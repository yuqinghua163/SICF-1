#include "spatialinvariantcolorfeature.h"
#include "omp.h"

#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_circle.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/features/fpfh.h>

ColorCoding::ColorCoding(const METHOD &_method)
    : method_(_method)//0xAA;//0xAA==170    HAMMING_GRAY8
{
    switch ( method_ )
    {
    case HAMMING_HSV422:
        coder_ = boost::shared_ptr<CodeBase> (new HSV422Code);
        code_type_ = CV_8UC1;
        break;
    case HAMMING_HSV655:
        coder_ = boost::shared_ptr<CodeBase> (new HSV655Code);
        code_type_ = CV_16UC1;
        break;
    case HAMMING_GRAY8:
        code_type_ = CV_8UC1;
        break;
    case HAMMING_GRAY16:
        code_type_ = CV_16UC1;
        break;
    default:
        code_type_ = CV_8UC1;
        break;
    }
}
ColorCoding::HSV422Code::HSV422Code()
{
    INVALID_CODE = 0b10101010;
    ///初始化编码表  用HSV值进行编码
    const code_t H_code[8] = {0x00,0x10,0x30,0x70,0xF0,0xE0,0xC0,0x80};//高四位
    const code_t S_code[3] = {0x00,0x04,0x0C};//第5、6位
    const code_t V_code[3] = {0x00,0x01,0x03};//末两位
    const code_t EMPITY_H = 0b01010000;

    std::vector<code_t> H_vec(8), S_vec(3), V_vec(3);
    std::memcpy( H_vec.data(), H_code, 8*sizeof(code_t) );
    std::memcpy( S_vec.data(), S_code, 3*sizeof(code_t) );
    std::memcpy( V_vec.data(), V_code, 3*sizeof(code_t) );
    initByHSV<uchar>( H_vec, S_vec, V_vec, EMPITY_H );
}

ColorCoding::HSV655Code::HSV655Code()
{
    ///初始化编码表  用HSV值进行编码
    const code_t H_code[12] ={0B0000000000000000,
                              0B0000010000000000,
                              0B0000110000000000,
                              0B0001110000000000,
                              0B0011110000000000,
                              0B0111110000000000,
                              0B1111110000000000,
                              0B1111100000000000,
                              0B1111000000000000,
                              0B1110000000000000,
                              0B1100000000000000,
                              0B1000000000000000};//high 6 bits
    const code_t S_code[6] = {0B0000000000000000,
                              0B0000000000100000,
                              0B0000000001100000,
                              0B0000000011100000,
                              0B0000000111100000,
                              0B0000001111100000};//middle 5 bits
    const code_t V_code[6] = {0B0000000000000000,
                              0B0000000000000001,
                              0B0000000000000011,
                              0B0000000000000111,
                              0B0000000000001111,
                              0B0000000000011111};//low 5 bits
    const code_t EMPITY_H  =  0B0101010000000000;
    INVALID_CODE           =  0B1010101010101010;

    std::vector<code_t> H_vec(12), S_vec(6), V_vec(6);
    std::memcpy( H_vec.data(), H_code, 12*sizeof(code_t) );
    std::memcpy( S_vec.data(), S_code, 6*sizeof(code_t) );
    std::memcpy( V_vec.data(), V_code, 6*sizeof(code_t) );
    initByHSV<u_int16_t>( H_vec, S_vec, V_vec, EMPITY_H );
}
template<typename code_t>
bool ColorCoding::CodeBase::initByHSV( const std::vector<code_t> &H_code, const std::vector<code_t> &S_code, const std::vector<code_t> &V_code, const code_t &EMPITY_H )
{
    size_t H_size = H_code.size();
    size_t S_size = S_code.size();
    size_t V_size = V_code.size();

    const uchar GRAY_THRESH_S = 50;//take it as gray if the S value is lower than this thresh
    const uchar GRAY_THRESH_V = 20;

    cv::Mat rgb2hsv_img(16,16*16,CV_8UC3);
    for(int r=0; r<16; r++)
    {
        uchar * p_img = rgb2hsv_img.data + r*rgb2hsv_img.step[0];
        for(int g=0; g<16; g++)
        for(int b=0; b<16; b++)
        {
            *p_img = b*16;
            *(p_img+1) = g*16;
            *(p_img+2) = r*16;
            p_img += rgb2hsv_img.step[1];
        }
    }
    cv::cvtColor(rgb2hsv_img, rgb2hsv_img, CV_BGR2HSV_FULL);
    for(int r=0; r<16; r++)
    {
        uchar * p_img = rgb2hsv_img.data + r*rgb2hsv_img.step[0];
        for(int g=0; g<16; g++)
        for(int b=0; b<16; b++)
        {
            uchar H = *p_img;
            uchar S = *(p_img+1);
            uchar V = *(p_img+2);
            if( S<=GRAY_THRESH_S || V<=GRAY_THRESH_V )
            {
                V = V*V_size/256;
                rgb2code_[r][g][b] = EMPITY_H | V_code[V];
            }
            else
            {
                H = H*H_size/256;
                V = V*V_size/256;
                S = (int)(S-GRAY_THRESH_S)*S_size/(256-GRAY_THRESH_S);
                rgb2code_[r][g][b] = H_code[H] | S_code[S] | V_code[V];
            }
            p_img += rgb2hsv_img.step[1];
        }
    }

    ///初始化code2rgba_
    code2rgba_.resize( 1<<(sizeof(code_t)*8), 0 );
    cv::Mat hsv_img(1,H_size*(S_size+1)*V_size,CV_8UC3);
    uchar H_show[H_size];
    uchar S_show[S_size+1];//empty H means S=0, so an extra S_show is added
    uchar V_show[V_size];
    for(int h=0; h<H_size; h++)
        H_show[h] = (h+0.5)*256/H_size;
    for(int s=0; s<S_size; s++)
        S_show[s] = (s+0.5)*(256-GRAY_THRESH_S)/S_size+GRAY_THRESH_S;
    S_show[S_size] = 0;
    for(int v=0; v<V_size; v++)
        V_show[v] = (v+0.5)*256/V_size;
    uchar *p_hsv_img = hsv_img.data;
    for(size_t H=0; H<H_size;   H++)
    for(size_t S=0; S<S_size+1; S++)
    for(size_t V=0; V<V_size;   V++)
    {
        *p_hsv_img     = H_show[H];
        *(p_hsv_img+1) = S_show[S];
        *(p_hsv_img+2) = V_show[V];
        p_hsv_img += hsv_img.step[1];
    }

    cv::Mat hsi2rgba( hsv_img.rows, hsv_img.cols, CV_8UC4 );
    cv::cvtColor(hsv_img, hsi2rgba, CV_HSV2BGR_FULL);
    uchar *p_hsi2rgba = hsi2rgba.data;
    for(size_t H=0; H<H_size;   H++)
    for(size_t S=0; S<S_size+1; S++)
    for(size_t V=0; V<V_size;   V++)
    {
        if(S!=S_size)
            code2rgba_[ H_code[H] | S_code[S] | V_code[V] ] = *(uint32_t*)p_hsi2rgba;
        else
            code2rgba_[ EMPITY_H | V_code[V] ] = *(uint32_t*)p_hsi2rgba;
        p_hsi2rgba += hsi2rgba.step[1];
    }
    return true;
}

int ColorCoding::encode(void *p_code, const uchar _r, const uchar _g, const uchar _b) const
{
    switch ( method_ )
    {
    case HAMMING_HSV422:
    {
        uchar &code = *(uchar*)p_code;
        code = coder_->rgb2code_[_r/16][_g/16][_b/16];
//        if( _V_MEAN != 128 )
//        {
//            int v = ((int)_r+(int)_g+(int)_b)/3 ;
//            if(      v < _V_MEAN-30 ) code = (code & 0xFC) | 0x00;
//            else if( v > _V_MEAN+30 ) code = (code & 0xFC) | 0x03;
//            else                      code = (code & 0xFC) | 0x01;
//        }
    }
        return sizeof(uchar);
    case HAMMING_HSV655:
    {
        u_int16_t &code = *(u_int16_t*)p_code;
        code = coder_->rgb2code_[_r/16][_g/16][_b/16];
    }
        return sizeof(u_int16_t);
    case HAMMING_GRAY8:
    {
        uchar &code = *(uchar*)p_code;
        int gray = ((int)_r+_g+_b)/3 * 9/256;//0~8
        code = (uchar)0xff >> (8-gray);
    }
        return 1;
    default:
        return 0;
    }
}

int ColorCoding::invalidCode( void* p_code ) const
{
    switch ( method_ )
    {
    case HAMMING_HSV422:
        *(uchar*)p_code = coder_->INVALID_CODE;
        return 1;
    case HAMMING_HSV655:
        *(u_int16_t*)p_code = coder_->INVALID_CODE;
        return 2;
    case HAMMING_GRAY8:
        *(uchar*)p_code = 0b10101010;
        return 1;
    default:
        return 0;
    }

}

uchar ColorCoding::rgb2IntCode(const uchar _r, const uchar _g, const uchar _b, const uchar _bit_length) const
{
    int chanel_range;
    if( _bit_length%3 == 0 )
        chanel_range = 1 << (_bit_length/3) ;
    else
        chanel_range = std::pow(2, _bit_length/3.0);
    //normalize r g b to [0,chanel_range)
    uchar r = (int)_r * chanel_range / 256;
    uchar g = (int)_g * chanel_range / 256;
    uchar b = (int)_b * chanel_range / 256;
    return r*chanel_range*chanel_range + g*chanel_range + b;
}

uint ColorCoding::machCode(void* _code1, void* _code2, const uint _cells) const
{
    uint dist = 0;
    uint invalid_cnt = 0;
    uint valid_cnt = 0;
    for(int i=0; i<_cells; i++)
    {
        switch ( method_ )
        {
        case HAMMING_HSV422:
        {
            typedef uchar code_type;
            const code_type *p1 = (code_type*)_code1 + i;
            const code_type *p2 = (code_type*)_code2 + i;
            if( *p1 == *p2 )
                dist += 0;
            else if( *p1 == coder_->INVALID_CODE || *p2 == coder_->INVALID_CODE )
                invalid_cnt ++;
            else
                dist += cv::normHamming( p1, p2, sizeof(code_type) );
        }
            break;
        case HAMMING_HSV655:
        {
            typedef u_int16_t code_type;
            const code_type *p1 = (code_type*)_code1 + i;
            const code_type *p2 = (code_type*)_code2 + i;
            if     ( *p1 == coder_->INVALID_CODE && *p2 == coder_->INVALID_CODE )
                dist += 0;
            else if( *p1 == coder_->INVALID_CODE || *p2 == coder_->INVALID_CODE )
                invalid_cnt ++;
            else
            {
                valid_cnt ++;
                dist += cv::normHamming( (uchar*)p1, (uchar*)p2, sizeof(code_type) );
            }
        }
            break;
        case HAMMING_GRAY8:

        default:
            return INFINITY;
        }
    }
    if( valid_cnt < invalid_cnt )
        return INFINITY;
    else
        return dist * _cells / (_cells-invalid_cnt);
}
uint32_t ColorCoding::decode(const void *_code) const
{
    switch ( method_ )
    {
    case HAMMING_HSV422:
        return coder_->code2rgba_[ *(uchar*)_code ];
    case HAMMING_HSV655:
        return coder_->code2rgba_[ *(u_int16_t*)_code ];
    case HAMMING_GRAY8:
    {
        uchar temp = cv::normHamming( (uchar*)_code, &temp, 1 ) *256/9;
        return (uint32_t)temp<<16 | (uint32_t)temp<<8 | (uint32_t)temp;
    }
    default:
        return 0;
    }
}

SpatialInvariantColorFeature::AnnularMask::AnnularMask(const uint &_patch_size, const uint &_angle_res, const uint &_dist_res)
    : PatchMask(_angle_res,_dist_res*_angle_res+1), DIST_RES(_dist_res)
{
    const uint RADIUS = _patch_size/2;
    TAN_ANGLE_THRESH.reserve(ANGLE_RES/4);
    for(size_t i=0; i<ANGLE_RES/4; i++)
        TAN_ANGLE_THRESH.push_back( tan( i*2*M_PI/ANGLE_RES ) );

    DIR_PER_CELL.resize(TOTAL_CELLS);
    DIR_PER_CELL[0].x =DIR_PER_CELL[0].y = 0;
    for(size_t d=0; d<DIST_RES; d++)
    for(size_t i=0; i<ANGLE_RES; i++)
    {
        DIR_PER_CELL[d*ANGLE_RES+i+1].x = cos( M_PI*2*(i+0.5)/ANGLE_RES );
        DIR_PER_CELL[d*ANGLE_RES+i+1].y = sin( M_PI*2*(i+0.5)/ANGLE_RES );
    }

    ///////初始化patch_mask_annular_
    mask_.create( _patch_size, _patch_size, CV_8UC1);
    std::vector<uint> dis_res_thresh2(DIST_RES+1);
    //保证每个单元格的面积相等则半径不等
    for(uint k=0; k<DIST_RES+1; k++)
    {
        double curr_r = sqrt(double(ANGLE_RES*k+1)/(ANGLE_RES*DIST_RES+1))*RADIUS;//共ANGLE_RES*DIST_RES+1个单元个(含中心一个)
        dis_res_thresh2[k] = (curr_r+0.5)*(curr_r+0.5);
    }
    //否则半径相等则面积不等
//    for(uint k=0; k<DIST_RES+1; k++)
//        dis_res_thresh2[k] = ( (k+0.5d)/(DIST_RES+0.5d)*RADIUS+0.5 ) * ( (k+0.5d)/(DIST_RES+0.5d)*RADIUS+0.5 );

    for(int i=0; i<mask_.rows; i++)
    for(int j=0; j<mask_.cols; j++)
    {
        int x = j - (int)RADIUS;
        int y = (int)RADIUS - i;
        if( (uint)x*x+y*y <= dis_res_thresh2[DIST_RES])
        {
            uint pos;
            if( (uint)x*x+y*y <= dis_res_thresh2[0] )
            {
                mask_.at<uchar>(i,j) = 0;
                continue;
            }
            for(pos=1; pos<TAN_ANGLE_THRESH.size() && abs(y)>abs(x)*TAN_ANGLE_THRESH[pos]; pos++)
                ;//取绝对值, 翻转到第1象限, 角度编号为 1~ANGLE_RES/4
            if     ( x>=0 && y>=0 ) pos--;                    // 第1象限: 0             ~ ANGLE_RES/4-1 ;
            else if( x<=0 && y>=0 ) pos = ANGLE_RES/2 - pos;  // 第2象限: ANGLE_RES/4-1 ~ ANGLE_RES/2-1 ;
            else if( x<=0 && y<=0 ) pos = ANGLE_RES/2 + pos-1;// 第3象限: ANGLE_RES/2-1 ~ ANGLE_RES/4*3-1;
            else if( x>=0 && y<=0 ) pos = ANGLE_RES - pos;    // 第4象限: ANGLE_RES/4*3-1 ~ ANGLE_RES-1;
            pos += 1;//跳过中心cell
            for(uint k=1; k<DIST_RES+1; k++)
            {
                if( (uint)x*x+y*y > dis_res_thresh2[k] )
                    pos += ANGLE_RES;// 1 ~ ANGLE_RES*DIST_RES;
                else break;
            }

            mask_.at<uchar>(i,j) = pos;
        }
        else
            mask_.at<uchar>(i,j) = BAD_CELL;
    }
//    patch_mask_annular_.at<uchar>(RADIUS,RADIUS) = BAD_CELL;

//    cv::Mat mask_show;
//    cv::resize(mask_,mask_show,cv::Size(_patch_size*4,_patch_size*4),0,0,cv::INTER_NEAREST);
//    cv::imshow("mask_show",mask_show*4);
//    char key = cv::waitKey();
//    if( key == 's' )
//        cv::imwrite("mask_show.bmp",mask_show);

}

const uchar
SpatialInvariantColorFeature::AnnularMask::getCellID(const uint& _xp, const uint& _yp, const uint& _main_angle_deg ) const
{
    uint dir_id = _main_angle_deg*ANGLE_RES/360;
    while( dir_id>=ANGLE_RES ) dir_id -= ANGLE_RES;
    const uchar& cell_id = mask_.at<uchar>(_yp,_xp);
    if( cell_id==BAD_CELL || cell_id==0 )
        return cell_id;
    else
    {
        if( (cell_id-1)%ANGLE_RES < dir_id )
            return cell_id + ANGLE_RES - dir_id;
        else
            return cell_id - dir_id;
    }
}

SpatialInvariantColorFeature::BeehiveMask::BeehiveMask(const uint &_patch_size, const uint &_layers, const uint &_angle_res)
    : PatchMask( _angle_res, 3*_layers*(_layers+1)+1 )//每个蜂窝是60度分辨率, 用两组蜂窝, 每个蜂窝有61个cell
{
    assert( ANGLE_RES%6==0 );//蜂窝六边形自带六个角度分辨率
    assert( _patch_size%2 ==1 );
    const uint BIG_SIZE = _patch_size*3;//考虑到需要对mask进行旋转，为了提高精度，用大图像来构造，最后再降采样
    const uint BIG_RADIUS = BIG_SIZE/2;
    /////初始化patch_mask_///////////////////////////////////////////////////////////
    /// 61个cells的蜂窝如图所示划分坐标系, j单位长度STEP_J=CELL_BORDER*1.5, i单位长度STEP_I=CELL_BORDER*sqrt(3)/2
    /// 则蜂窝总宽STEP_J*10, 高GSTEP_I*18
    /// 61个cells的编号原则为从内层到外层, 逆时针一到四象限
    /*/ 图示为第1象限的1/4个蜂窝的示意图, cell中数字为编号, 也是patch_mask_beehive_中存储的数字
    ///    0     1      2      3      4      5
    ///   __________________________________ j
    /// 0 |0  /     \  10   /     \  43   /
    ///   |__/       \_____/       \_____/  ff
    /// 1 |  \   2   /     \  23   /     \
    ///   |   \_____/       \_____/       \ ff
    /// 2 |1  /     \   9   /     \  41   /
    ///   |__/       \_____/       \_____/  ff
    /// 3 |  \   8   /     \  22   /     \
    ///   |   \_____/       \_____/       \ ff
    /// 4 |7  /     \  21   /     \  41   /
    ///   |__/       \_____/       \_____/  ff
    /// 5 |  \  20   /     \  40   /
    ///   |   \_____/       \_____/  ff     ff
    /// 6 |19 /     \  39   /
    ///   |__/       \_____/  ff     ff     ff
    /// 7 |  \  38   /
    ///   |   \_____/  ff     ff     ff     ff
    /// 8 |37 /
    ///   |__/  ff     ff     ff     ff     ff
    /// 9 i
    /*/
    cv::Mat mask0  = cv::Mat( BIG_SIZE, BIG_SIZE, CV_8UC1, BAD_CELL);
    const double CELL_BORDER =     BIG_SIZE/((_layers*2+1)*sqrt(3));    //六边形边长
    const double STEP_J  =      CELL_BORDER*1.5;
    const double STEP_I =       CELL_BORDER*sqrt(3)/2 ;
    const double TEMP_1      =      (STEP_J*STEP_J - STEP_I*STEP_I)/2.0;
    const double TEMP_2      =      (STEP_J*STEP_J + STEP_I*STEP_I)/2.0;
    std::vector<uint> CELLS_PER_RING( _layers+1 );
    std::vector<uint> START_PER_RING( _layers+1 );
    CELLS_PER_RING[0] = 1;
    START_PER_RING[0] = 0;
    for(uint i=1; i<=_layers; i++)
    {
        CELLS_PER_RING[i] = i*6;
        START_PER_RING[i] = START_PER_RING[i-1] + CELLS_PER_RING[i-1];
    }

    for(int i=0; i<=(int)BIG_RADIUS; i++)
    for(int j=0; j<=(int)BIG_RADIUS; j++)
    {
        int cell_j = (int)(j/STEP_J);
        int cell_i = (int)(i/STEP_I);
        if((cell_j+cell_i)&1)
        {
            if((j-cell_j*STEP_J)*STEP_J-(i-cell_i*STEP_I)*STEP_I > TEMP_1) cell_j++;
        }
        else
            if((j-cell_j*STEP_J)*STEP_J+(i-cell_i*STEP_I)*STEP_I > TEMP_2) cell_j++;
        if((cell_j+cell_i)&1)
            cell_i++;

        uchar ring_id;//层数/环数
        for( ring_id=0; cell_j>ring_id || cell_j+cell_i>ring_id*2; ring_id++)
            ;
        if( ring_id<=_layers )//共四层, 以外的像素不要
        {
            uchar pos_temp = BAD_CELL;//当前象限, 当前环中的临时编号0~CELLS_PER_RING/4
            if( cell_j==ring_id ) pos_temp = CELLS_PER_RING[ring_id]/4 - cell_i/2;//竖着的几个格
            else pos_temp = cell_j;//斜线的几个格
            mask0.at<uchar>(BIG_RADIUS-j,BIG_RADIUS+i) = START_PER_RING[ring_id] + pos_temp;                            //第1象限
            mask0.at<uchar>(BIG_RADIUS-j,BIG_RADIUS-i) = START_PER_RING[ring_id] - pos_temp + CELLS_PER_RING[ring_id]/2;//第2象限
            mask0.at<uchar>(BIG_RADIUS+j,BIG_RADIUS-i) = START_PER_RING[ring_id] + pos_temp + CELLS_PER_RING[ring_id]/2;//第3象限
            mask0.at<uchar>(BIG_RADIUS+j,BIG_RADIUS+i) = START_PER_RING[ring_id] - pos_temp + CELLS_PER_RING[ring_id];  //第4象限
            if( pos_temp==0 )//第4象限最后一个cell归属于第1象限
                mask0.at<uchar>(BIG_RADIUS+j,BIG_RADIUS+i) = START_PER_RING[ring_id] - pos_temp;//第4象限
        }
    }

    masks_.resize( ANGLE_RES/6 );//蜂窝六边形自带六个角度分辨率
    double angle_deg = 0;
    cv::Mat mat_temp(BIG_SIZE, BIG_SIZE, CV_8UC1);
    for( std::vector<cv::Mat>::iterator it = masks_.begin(); it != masks_.end(); it++, angle_deg += 360/ANGLE_RES )
    {
        const cv::Mat &rotate_mat = cv::getRotationMatrix2D( cv::Point2f(BIG_RADIUS,BIG_RADIUS), angle_deg, 1 );
        cv::warpAffine( mask0, mat_temp, rotate_mat, cv::Size(BIG_SIZE,BIG_SIZE), cv::INTER_NEAREST, cv::BORDER_CONSTANT, CV_RGB(BAD_CELL,BAD_CELL,BAD_CELL) );
        cv::resize(mat_temp,*it,cv::Size(_patch_size,_patch_size),0,0,cv::INTER_NEAREST);
    }

    rotateCellID_ = cv::Mat::zeros( 6, TOTAL_CELLS, CV_8UC1 );//6 angles of an beehive
    for(uint a=0; a<6; a++)
    for(uint pos=0; pos<TOTAL_CELLS; pos++)
    {
        uint ring_id = 0;
        for( ring_id=0; ring_id<_layers && pos>=START_PER_RING[ring_id+1]; ring_id++ )
            ;
        uint pos_inc = CELLS_PER_RING[ring_id]/(ANGLE_RES/2) * a;
        if( pos < START_PER_RING[ring_id] + pos_inc )
            rotateCellID_.at<uchar>(a,pos) = pos + CELLS_PER_RING[ring_id] - pos_inc;
        else
            rotateCellID_.at<uchar>(a,pos) = pos - pos_inc;
    }


//    for( std::vector<cv::Mat>::iterator it = masks_.begin(); it != masks_.end(); it++ )
//    {
//        cv::Mat mask_show;
//        cv::resize(*it,mask_show,cv::Size(_patch_size*4,_patch_size*4),0,0,cv::INTER_NEAREST);
//        cv::imshow("mask_show",mask_show*4);
//        char key = cv::waitKey();
//        if( key == 's' )
//            cv::imwrite("mask_show.bmp",mask_show);
//    }
}
const uchar
SpatialInvariantColorFeature::BeehiveMask::getCellID(const uint& _xp, const uint& _yp, const uint& _main_angle_deg ) const
{
    uint dir_id = _main_angle_deg*ANGLE_RES/360.0 + 0.5;
    while( dir_id>=ANGLE_RES ) dir_id -= ANGLE_RES;
    uchar cell_id = masks_[dir_id%(ANGLE_RES/6)].at<uchar>(_yp,_xp);
    if( cell_id != BAD_CELL )
        return rotateCellID_.at<uchar>(dir_id/(ANGLE_RES/6),cell_id);
    else
        return BAD_CELL;
}

SpatialInvariantColorFeature::SpatialInvariantColorFeature(const uint _max_keypoints, const uint _patch_radius, const DESCRIPTOR_TYPE _method)
    : PATCH_SIZE(_patch_radius*2+1), MAX_KEYPOINTS(_max_keypoints)
{
    DRAW_IMAG = true;
    SPATIAL_R = 70;//117 70
    annular_mask_ = boost::shared_ptr<AnnularMask> ( new AnnularMask( PATCH_SIZE, 36, 4 ) );
    beehive_mask_ = boost::shared_ptr<BeehiveMask> ( new BeehiveMask( PATCH_SIZE, 6 ) );
    patch_type_ = _method;// D_TYPE_BEEHIVE D_TYPE_BEEHIVE_NOOB D_TYPE_BRIEF D_TYPE_ORB D_TYPE_SURF D_TYPE_HISTOGRAM
    if( patch_type_ == D_TYPE_ANNULAR )
        patch_mask_ = annular_mask_;
    else//PATCH_TYPE_BEEHIVE
        patch_mask_ = beehive_mask_;

    //RGB相机标定模型
    ///kinect2
    //QHD: fx=517.704829, cx=476.8145455, fy=518.132948, cy=275.4860225,
    //530.7519923402057, 0.0, 478.15372152637906, 0.0, 529.1110630882142, 263.21561548634605
    camera_fx = 530.7519923402057;
    camera_fy = 529.1110630882142;
    camera_cx = 478.15372152637906;
    camera_cy = 263.21561548634605;
    ///kinect1
//    camera_fx = 595;
//    camera_fy = 595;
    camera_cx = (640-1)/2.0;
    camera_cy = (480-1)/2.0;

    _1_camera_fx   =   1.0/camera_fx;
    _1_camera_fy   =   1.0/camera_fy;
    _256_camera_fx = 256.0/camera_fx;
    _256_camera_fy = 256.0/camera_fy;

    keypoints_filtered_.reserve( MAX_KEYPOINTS );
    keypoints_3D_.reserve(MAX_KEYPOINTS);
    features_show_   .create( 10*PATCH_SIZE, 10*PATCH_SIZE, CV_8UC4);//显示10*10个特征图像片段
    features_restore_.create( 10*PATCH_SIZE, 10*PATCH_SIZE, CV_8UC4);//显示10*10个特征图像片段
}

uint
SpatialInvariantColorFeature::extractFeaturePatch(const cv::Point& _pt, cv::Point3f &_pt3d, cv::Mat& _feature_patch, cv::Vec4d &_plane_coef , double &_plane_err, const uint &SPATIAL_RADIUS)
{//_plan_coef: Ax+By+Cz=D; return num of pixels

    const bool SHOW_TIME_INFO = false;
    timeval time0, time1, time2;
    if(SHOW_TIME_INFO)
    {
        gettimeofday(&time1,NULL);
        time0 = time1;
    }

    assert( _feature_patch.cols == PATCH_SIZE && _feature_patch.rows == PATCH_SIZE );
    assert( _feature_patch.channels()==4 );

    _feature_patch.setTo(0);
    uint REGION_GROW_RADIUS = 50;        //每个特征片段最大ROI半径,用于区域增长单位:像素
    const uint width = REGION_GROW_RADIUS*2+1;
    const uint r_nearest = 3;//r_roi/2;//首先用一个较小的临域来估计前景的深度范围，如果直接用大ROI的话，有可能会检测到其他更前的前景
    cv::Mat nearest_roi(depth_16U_, cv::Rect(_pt.x-r_nearest, _pt.y-r_nearest, r_nearest*2+1, r_nearest*2+1));//without copy data,only new header
    const uint r_plane = r_nearest*2;//用一个较小的临域来进行平面拟合

    ///截取特征点处的ROI
    const uint THRESH_NOISE_POINTS = 1;//如果某一个深度的点，总点数少于此，则认为是噪声
    ///构建小范围深度直方图(量化至1280级，分辨率为1cm)
    const uint HIST_RESOLUTION = 10;//unit:mm
    const uint DEPTH_MAX = 12800;//unit:mm
    std::vector<ushort> depth_8U_hist(DEPTH_MAX/HIST_RESOLUTION,0);
    uint min_depth_8U = DEPTH_MAX/HIST_RESOLUTION-1;
    uint max_depth_8U = 0;
    uchar *pdata;
    uint point_num = 0;
    uint depth_uchar;
    for(size_t i =0; i<r_nearest*2+1; i++)
    {
        pdata = nearest_roi.data + i*nearest_roi.step[0];
        for(size_t j=0; j<r_nearest*2+1; j++)
        {
            const u_int16_t &depth_current = *(u_int16_t*)pdata;
            depth_uchar = depth_current>=DEPTH_MAX ? 0 : depth_current/HIST_RESOLUTION;
            if( depth_uchar != 0 )
            {
                point_num ++;
                depth_8U_hist[depth_uchar]++;
                if( depth_uchar<min_depth_8U && depth_uchar!=0 ) min_depth_8U = depth_uchar;
                if( depth_uchar>max_depth_8U ) max_depth_8U = depth_uchar;
            }
            pdata += nearest_roi.step[1];
        }
    }
    if( point_num<THRESH_NOISE_POINTS*2 )
        return 0;

    ///求取小范围前景的深度范围
    for( ; depth_8U_hist[min_depth_8U]<THRESH_NOISE_POINTS && min_depth_8U!=max_depth_8U; min_depth_8U++) ;
    for( ; depth_8U_hist[max_depth_8U]<THRESH_NOISE_POINTS && min_depth_8U!=max_depth_8U; max_depth_8U--) ;
    uint front_depth_8U_thresh = min_depth_8U;
    uint gap_width = 0;
    for( size_t i=min_depth_8U+1; i<=max_depth_8U; i++ )
    {
        if( depth_8U_hist[i]<THRESH_NOISE_POINTS )
            gap_width ++;
        else if(gap_width*HIST_RESOLUTION<SPATIAL_R)
            front_depth_8U_thresh = i;
    }
    float front_thresh_max = (float)front_depth_8U_thresh*HIST_RESOLUTION+HIST_RESOLUTION-1;
    float front_thresh_min = (float)min_depth_8U*HIST_RESOLUTION;
    front_thresh_max += SPATIAL_R/2;
    front_thresh_min -= SPATIAL_R/2;

    ///提取小范围内前景点用于平面拟合,并记录下前景点用于区域增长
    cv::Mat plane_points = cv::Mat( (r_plane*2+1)*(r_plane*2+1), 3, CV_64F);//所有前景点,预留足够空间，之后会截取ROI来使用
    uchar *p_plane_points = plane_points.data;
    std::vector< pcl::PointXYZRGB, Eigen::aligned_allocator<pcl::PointXYZRGB> >::iterator it_cloud = cloud_->points.begin();
    enum FrontFlag
    {
        UN_KNOWN = 0,
        POSITIVE = 255,
        NEGATIVE = 127
    };
    cv::Mat front_flag = cv::Mat::zeros( width, width, CV_8U);//用于区域增长的标志位
    front_flag.setTo(UN_KNOWN);
//    cv::Mat front_flag_palne(front_flag, cv::Rect(r_roi-r_plane, r_roi-r_plane, r_plane*2+1, r_plane*2+1));//without copy data,only new header
    std::vector< cv::Point3i > front_flag_vector;//区域增长的标志位所对应的像素点(i,j,d)
    front_flag_vector.reserve( width*width );
    uint front_points_num = 0;
    double front_centroid_x = 0;//所有前景点的均值，用于求SVD之前先无偏
    double front_centroid_y = 0;
    double front_centroid_z = 0;
    for(size_t h =0; h<r_plane*2+1; h++)
    {
        it_cloud = cloud_->points.begin() + (_pt.y-r_plane+h)*cloud_->width + _pt.x-r_plane;
        uchar * p_front_flag = front_flag.data + (REGION_GROW_RADIUS-r_plane+h)*front_flag.step[0] + (REGION_GROW_RADIUS-r_plane);
        for(size_t w=0; w<r_plane*2+1; w++)
        {
            const float &depth_current = it_cloud->z;
            if( std::isfinite(depth_current) && depth_current!=0 && depth_current<=front_thresh_max && depth_current>=front_thresh_min)
            {
                ((double*)p_plane_points)[0] = it_cloud->x;
                ((double*)p_plane_points)[1] = it_cloud->y;
                ((double*)p_plane_points)[2] = depth_current;
                front_centroid_x += it_cloud->x;
                front_centroid_y += it_cloud->y;
                front_centroid_z += depth_current;
                front_points_num++;
                p_plane_points  += plane_points.step[0];

                *p_front_flag = POSITIVE;
                front_flag_vector.push_back( cv::Point3i(w+REGION_GROW_RADIUS-r_plane,h+REGION_GROW_RADIUS-r_plane,depth_current) );
//                std::cout<<":"<<w+r_roi-r_plane<<" "<<h+r_roi-r_plane<<" "<<depth_current<<std::endl;
            }
            else
                *p_front_flag = NEGATIVE;
            it_cloud++;
            p_front_flag ++;
        }
    }
    if( front_points_num<2 )
        return 0;
    front_centroid_x /= front_points_num;
    front_centroid_y /= front_points_num;
    front_centroid_z /= front_points_num;
    cv::Mat plane_points_ROI(plane_points, cv::Rect(0, 0, 3, front_points_num));//without copy data,only new header

    if(SHOW_TIME_INFO)
    {
        gettimeofday(&time2,NULL);
        std::cout << " 提取前景 "  << (time2.tv_sec-time1.tv_sec)*1000+(time2.tv_usec-time1.tv_usec)/1000.0;
        time1 = time2;
    }
    ///用SVD求取前景的平面拟合。
    //正常的SVD求取平面的方法，需要每个点都减去中心均值点，相当于无偏的过程。这本质上是要求将平面过原点。
    p_plane_points = plane_points.data;
    for( size_t i=0; i<front_points_num; i++)
    {
        ((double*)p_plane_points)[0] -= front_centroid_x;
        ((double*)p_plane_points)[1] -= front_centroid_y;
        ((double*)p_plane_points)[2] -= front_centroid_z;
        p_plane_points += plane_points.step[0];
    }
    cv::Mat U, S, VT; //A=U*S*VT; S为特征值，VT为特征向量。opencv中解得的是V的转置，matlab得到的直接是V
    cv::SVDecomp(plane_points_ROI,S,U,VT);//,cv::SVD::FULL_UV);  //后面的FULL_UV表示把U和VT补充成单位正交方阵;
//    std::cout << "S="  << std::endl << S  << std::endl;
//    std::cout << "VT=" << std::endl << VT << std::endl;
    //最小的特征向量即为平面的法向向量，即平面上点与之正交: ([x,y,z]-front_centroid)*VT(3)=0
    double * axis_x = (double*)(VT.data+VT.step[0]*0);
    double * axis_y = (double*)(VT.data+VT.step[0]*1);
    double * axis_z = (double*)(VT.data+VT.step[0]*2);

    pcl::PointXYZRGB searchPoint;
    std::vector<int> pointIndicesOut;
    std::vector<float> pointRadiusSquaredDistance;
    searchPoint.z = (front_thresh_max+front_thresh_min)/2;
    searchPoint.x = (_pt.x-camera_cx+0.5) * _1_camera_fx * searchPoint.z;
    searchPoint.y = (_pt.y-camera_cy+0.5) * _1_camera_fy * searchPoint.z;
//    double rad = 0.100; // search radius
//    kdtree_.radiusSearch( searchPoint, rad, pointIndicesOut, pointRadiusSquaredDistance );
    kdtree_.nearestKSearch( searchPoint, 1, pointIndicesOut, pointRadiusSquaredDistance );
    pcl::Normal &point_norm = normals_->at( pointIndicesOut[0] );
    axis_z[0] = point_norm.normal[0];
    axis_z[1] = point_norm.normal[1];
    axis_z[2] = point_norm.normal[2];

    std::memcpy( _plane_coef.val, axis_z, sizeof(double)*3 );
    _plane_coef[3] = front_centroid_x*axis_z[0]+front_centroid_y*axis_z[1]+front_centroid_z*axis_z[2];
    _plane_err = ( (double*)S.data)[2] / hypot( ((double*)S.data)[0], ((double*)S.data)[1] );



    ///特征点三维坐标
    const pcl::PointXYZRGB &point = cloud_->at(_pt.x,_pt.y);
    if( point.z!=0  && point.z<front_thresh_max )
    {
        _pt3d.z = point.z;
        _pt3d.x = (_pt.x-camera_cx + 0.5) * _1_camera_fx * point.z;
        _pt3d.y = (_pt.y-camera_cy + 0.5) * _1_camera_fy * point.z;
    }
    else
    {
        float pt_xn = (_pt.x-camera_cx + 0.5) * _1_camera_fx;
        float pt_yn = (_pt.y-camera_cy + 0.5) * _1_camera_fy;
        _pt3d.z = _plane_coef[3] / ( pt_xn*_plane_coef[0]+pt_yn*_plane_coef[1]+_plane_coef[2] );
        _pt3d.x = pt_xn * _pt3d.z;
        _pt3d.y = pt_yn * _pt3d.z;
    }

    if(SHOW_TIME_INFO)
    {
        gettimeofday(&time2,NULL);
        std::cout << " SVD "  << (time2.tv_sec-time1.tv_sec)*1000+(time2.tv_usec-time1.tv_usec)/1000.0;
        time1 = time2;
    }

    ///区域增长
    const uint THRESH_GAP_GROW = SPATIAL_R;//单位mm,大于此宽度的断层，则认为是前、后景分离
    for( std::vector< cv::Point3i >::iterator it_flag_point = front_flag_vector.begin(); it_flag_point!=front_flag_vector.end(); it_flag_point++)
    {
        const int& xp = it_flag_point->x;
        const int& yp = it_flag_point->y;
        const int& d  = it_flag_point->z;
        uchar *p_front_flag = front_flag.data + yp*front_flag.step[0] + xp*front_flag.step[1];
        uchar *p_depth_16U  = depth_16U_.data + (_pt.y-REGION_GROW_RADIUS+yp)*depth_16U_.step[0] + (_pt.x-REGION_GROW_RADIUS+xp)*depth_16U_.step[1];
        if( yp>1 )//上临域
        {
            uchar & cur_flag = *(p_front_flag-front_flag.step[0]);
            if( cur_flag == UN_KNOWN )
            {
                u_int16_t & cur_depth = *(u_int16_t*)(p_depth_16U-depth_16U_.step[0]);
                if( !std::isnan( cur_depth ) )
                {
                    uint gap_temp = abs(d-cur_depth);
                    if( gap_temp < THRESH_GAP_GROW/2 )
                    {
                        cur_flag = POSITIVE;
                        front_flag_vector.push_back( cv::Point3i(xp,yp-1,cur_depth) );
                        front_points_num ++;
                    }
                    else if( gap_temp > THRESH_GAP_GROW )
                        cur_flag = NEGATIVE;
                }
                else
                    cur_flag = NEGATIVE;
            }
        }
        if( yp+1<(int)width )//下临域
        {
            uchar & cur_flag = *(p_front_flag+front_flag.step[0]);
            if( cur_flag == UN_KNOWN )
            {
                u_int16_t & cur_depth = *(u_int16_t*)(p_depth_16U+depth_16U_.step[0]);
                if( !std::isnan( cur_depth ) )
                {
                    uint gap_temp = abs(d-cur_depth);
                    if( gap_temp < THRESH_GAP_GROW/2 )
                    {
                        cur_flag = POSITIVE;
                        front_flag_vector.push_back( cv::Point3i(xp,yp+1,cur_depth) );
                        front_points_num ++;
                    }
                    else if( gap_temp > THRESH_GAP_GROW )
                        cur_flag = NEGATIVE;
                }
                else
                    cur_flag = NEGATIVE;
            }
        }

        if( xp>1 )//左临域
        {
            uchar & cur_flag = *(p_front_flag-front_flag.step[1]);
            if( cur_flag == UN_KNOWN )
            {
                u_int16_t & cur_depth = *(u_int16_t*)(p_depth_16U-depth_16U_.step[1]);
                if( !std::isnan( cur_depth ) )
                {
                    uint gap_temp = abs(d-cur_depth);
                    if( gap_temp < THRESH_GAP_GROW/2 )
                    {
                        cur_flag = POSITIVE;
                        front_flag_vector.push_back( cv::Point3i(xp-1,yp,cur_depth) );
                        front_points_num ++;
                    }
                    else if( gap_temp > THRESH_GAP_GROW )
                        cur_flag = NEGATIVE;
                }
                else
                    cur_flag = NEGATIVE;
            }
        }
        if( xp+1<(int)width )//右临域
        {
            uchar & cur_flag = *(p_front_flag+front_flag.step[1]);
            if( cur_flag == UN_KNOWN )
            {
                u_int16_t & cur_depth = *(u_int16_t*)(p_depth_16U+depth_16U_.step[1]);
                if( !std::isnan( cur_depth ) )
                {
                    uint gap_temp = abs(d-cur_depth);
                    if( gap_temp < THRESH_GAP_GROW/2 )
                    {
                        cur_flag = POSITIVE;
                        front_flag_vector.push_back( cv::Point3i(xp+1,yp,cur_depth) );
                        front_points_num ++;
                    }
                    else if( gap_temp > THRESH_GAP_GROW )
                        cur_flag = NEGATIVE;
                }
                else
                    cur_flag = NEGATIVE;
            }
        }
    }

    if(SHOW_TIME_INFO)
    {
        gettimeofday(&time2,NULL);
        std::cout << " 区域增长 "  << (time2.tv_sec-time1.tv_sec)*1000+(time2.tv_usec-time1.tv_usec)/1000.0;
        time1 = time2;
    }

    ///将前景点映射到正视平面上(x2d,y2d)
    //U矩阵乘以S, 实际上就是各个点在平面上的坐标, 但是坐标原点是质心, 需要平移到新的中心
    cv::Vec3f center( _pt3d.x-front_centroid_x, _pt3d.y-front_centroid_y, _pt3d.z-front_centroid_z );
    cv::Vec3d proj_center;//新的中心
    proj_center[0] = center[0]*axis_x[0] + center[1]*axis_x[1] + center[2]*axis_x[2];
    proj_center[1] = center[0]*axis_y[0] + center[1]*axis_y[1] + center[2]*axis_y[2];
    proj_center[2] = center[0]*axis_z[0] + center[1]*axis_z[1] + center[2]*axis_z[2];
    double W2P_RATIO = double(PATCH_SIZE/2)/(double)SPATIAL_RADIUS;//空间尺度mm到图像片像素的变换比例
    std::vector<cv::Point2i> pixel_radius_table;    //某个像素在某个深度下所对应的图像片像素尺度(因此patch_map_大小有关)，用于视角变换时插值放大, 输入为 深度值/100
    pixel_radius_table.resize(128);
    for(uint i=0; i<pixel_radius_table.size(); i++)
    {
        pixel_radius_table[i].x = int( i*100*_1_camera_fx * W2P_RATIO + 1 ) /2;
        pixel_radius_table[i].y = int( i*100*_1_camera_fy * W2P_RATIO + 1 ) /2;
    }
    double xn, yn, zn;
    for(size_t h =0; h<width; h++)
    {
        it_cloud = cloud_->points.begin() + (_pt.y-REGION_GROW_RADIUS+h)*cloud_->width + _pt.x-REGION_GROW_RADIUS;
        uchar *p_front_flag = front_flag.data + h*front_flag.step[0];
        for(size_t w=0; w<width; w++)
        {
            if( *p_front_flag==POSITIVE )
            {
                xn = it_cloud->x - front_centroid_x;
                yn = it_cloud->y - front_centroid_y;
                zn = it_cloud->z - front_centroid_z;
                double proj_x = xn*axis_x[0] + yn*axis_x[1] + zn*axis_x[2] - proj_center[0];
                double proj_y = xn*axis_y[0] + yn*axis_y[1] + zn*axis_y[2] - proj_center[1];
                double proj_z = xn*axis_z[0] + yn*axis_z[1] + zn*axis_z[2] - proj_center[2];
                if( _plane_coef[2]<0 ) proj_x = -proj_x;//防止反向视角
//                if( fabs(proj_z-proj_center[2]) > THRESH_GAP_GROW*1.4 )
//                    continue;

                int x_pixel = proj_x * W2P_RATIO + 0.5;//计算它在正视图像中的坐标
                int y_pixel = proj_y * W2P_RATIO + 0.5;//计算它在正视图像中的坐标
                if( x_pixel*x_pixel + y_pixel*y_pixel > (PATCH_SIZE/2)*(PATCH_SIZE/2) )
                    continue;
                x_pixel += PATCH_SIZE/2, y_pixel += PATCH_SIZE/2;
                uint depth_temp = std::min( size_t(it_cloud->z)+99, pixel_radius_table.size()*100-1 );
                int& pixel_size_rx = pixel_radius_table[depth_temp/100].x;//当前像素点所占据的像素面积（由于远近不同，空间尺度也不同）
                int& pixel_size_ry = pixel_radius_table[depth_temp/100].y;//当前像素点所占据的像素面积（由于远近不同，空间尺度也不同）
                int min_px = x_pixel-pixel_size_rx-1, max_px = x_pixel+pixel_size_rx+1;//为了区域增长, 再额外增大一个像素
                int min_py = y_pixel-pixel_size_ry-1, max_py = y_pixel+pixel_size_ry+1;//为了区域增长, 再额外增大一个像素
                for(int y=min_py; y<=max_py; y++)
                if( y>=0 && y<(int)PATCH_SIZE )
                {
                    uchar *p_patch = _feature_patch.data + y*_feature_patch.step[0] + min_px*_feature_patch.step[1];
                    for(int x=min_px; x<=max_px; x++)
                    {
                        if( x>=0 && x<(int)PATCH_SIZE  )
                        if( !*(uint32_t*)p_patch || !( x==min_px || x==max_px || y==min_py || y==max_py) )
                            *(uint32_t*)p_patch = it_cloud->rgba;
                        p_patch += _feature_patch.step[1];
                    }
                }
            }
            it_cloud++;
            p_front_flag ++;
        }
    }
/// Blur and Grow is really necessary here!!!
    cv::Mat feature_patch_blur = _feature_patch.clone();
    static const cv::Mat kernel_d = cv::getGaussianKernel( PATCH_SIZE/10, 10, CV_64F );
    cv::Mat kernel = cv::Mat( kernel_d.rows, kernel_d.cols, CV_8U);
    kernel_d.convertTo( kernel, CV_8U, 255 );

    for(int h=0; h<feature_patch_blur.rows; h++ )
    {
    uchar *p_data = feature_patch_blur.data + h*feature_patch_blur.step[0];
    for(int w=0; w<feature_patch_blur.cols; w++)
    {
        for(int hk=0; hk<kernel.rows; hk++)
        {
        uchar *p_kernel =  kernel.data + hk*kernel.step[0];
        for(int wk=0; wk<kernel.cols; wk++)
        {
             p_kernel += kernel.step[1];
        }
        }
        p_data += feature_patch_blur.step[1];
    }
    }

    if(SHOW_TIME_INFO)
    {
        gettimeofday(&time2,NULL);
        std::cout << " 映射 "  << (time2.tv_sec-time1.tv_sec)*1000+(time2.tv_usec-time1.tv_usec)/1000.0;
        std::cout <<" 共: " << (time2.tv_sec-time0.tv_sec)*1000+(time2.tv_usec-time0.tv_usec)/1000.0<<"ms"<<std::endl;
    }
    return front_points_num;
}

uint SpatialInvariantColorFeature::calcPt6d(const cv::Point& _pt, cv::Point3f &_pt3d, cv::Vec4d &_plane_coef, double &_plane_err )
{//_plan_coef: Ax+By+Cz=D; return num of pixels

    const bool SHOW_TIME_INFO = false;
    timeval time0, timel, timen;
    if(SHOW_TIME_INFO)
    {
        gettimeofday(&timel,NULL);
        time0 = timel;
    }

    const uint r_nearest = 6;//r_roi/2;//首先用一个较小的临域来估计前景的深度范围，如果直接用大ROI的话，有可能会检测到其他更前的前景
    cv::Mat nearest_roi(depth_16U_, cv::Rect(_pt.x-r_nearest, _pt.y-r_nearest, r_nearest*2+1, r_nearest*2+1));//without copy data,only new header

    ///截取特征点处的ROI
    const uint THRESH_NOISE_POINTS = 1;//如果某一个深度的点，总点数少于此，则认为是噪声
    ///构建小范围深度直方图(量化至1280级，分辨率为1cm)
    const uint HIST_RESOLUTION = 10;//unit:mm
    const uint DEPTH_MAX = 12800;//unit:mm
    std::vector<ushort> depth_8U_hist(DEPTH_MAX/HIST_RESOLUTION,0);
    uint min_depth_8U = DEPTH_MAX/HIST_RESOLUTION-1;
    uint max_depth_8U = 0;
    uchar *pdata;
    uint point_num = 0;
    uint depth_uchar;
    for(size_t i =0; i<r_nearest*2+1; i++)
    {
        pdata = nearest_roi.data + i*nearest_roi.step[0];
        for(size_t j=0; j<r_nearest*2+1; j++)
        {
            const u_int16_t &depth_current = *(u_int16_t*)pdata;
            depth_uchar = depth_current>=DEPTH_MAX ? 0 : depth_current/HIST_RESOLUTION;
            if( depth_uchar != 0 )
            {
                point_num ++;
                depth_8U_hist[depth_uchar]++;
                if( depth_uchar<min_depth_8U && depth_uchar!=0 ) min_depth_8U = depth_uchar;
                if( depth_uchar>max_depth_8U ) max_depth_8U = depth_uchar;
            }
            pdata += nearest_roi.step[1];
        }
    }
    if( point_num<THRESH_NOISE_POINTS*2 )
        return 0;

    ///求取小范围前景的深度范围
    for( ; depth_8U_hist[min_depth_8U]<THRESH_NOISE_POINTS && min_depth_8U!=max_depth_8U; min_depth_8U++) ;
    for( ; depth_8U_hist[max_depth_8U]<THRESH_NOISE_POINTS && min_depth_8U!=max_depth_8U; max_depth_8U--) ;
    uint front_depth_8U_thresh = min_depth_8U;
    uint gap_width = 0;
    for( size_t i=min_depth_8U+1; i<=max_depth_8U; i++ )
    {
        if( depth_8U_hist[i]<THRESH_NOISE_POINTS )
            gap_width ++;
        else if(gap_width*HIST_RESOLUTION<SPATIAL_R)
            front_depth_8U_thresh = i;
    }
    float front_thresh_max = (float)front_depth_8U_thresh*HIST_RESOLUTION+HIST_RESOLUTION-1;
    float front_thresh_min = (float)min_depth_8U*HIST_RESOLUTION;
    front_thresh_max += SPATIAL_R/2;
    front_thresh_min -= SPATIAL_R/2;

    ///surch for nearest point which is on the feture plane
    pcl::PointXYZRGB surface_pt;
    bool pt_found = false;
    const pcl::PointXYZRGB &pt = cloud_->at(_pt.x, _pt.y);
    if( pt.z!=0 && pt.z<front_thresh_max )
    {
        pt_found = true;
        surface_pt.x = pt.x;
        surface_pt.y = pt.y;
        surface_pt.z = pt.z;
    }
    for(int r=1; r<=r_nearest && !pt_found; r++)//distance
    for(int side=0; side<4    && !pt_found; side++)//four sides: clockwise
    for(int i=-r; i<r         && !pt_found; i++)
    {
        int x_off, y_off;
        if     (side==0) x_off = i, y_off=-r;//up       //  0 0 1
        else if(side==1) x_off = r, y_off= i;//right    //  3 * 1
        else if(side==2) x_off =-i, y_off= r;//down     //  3 2 2
        else if(side==3) x_off =-r, y_off=-i;//left
        const pcl::PointXYZRGB &pt = cloud_->at(_pt.x+x_off, _pt.y+y_off);
        if( pt.z!=0 && pt.z<front_thresh_max )
        {
            pt_found = true;
            surface_pt.x = pt.x;
            surface_pt.y = pt.y;
            surface_pt.z = pt.z;
        }
    }
    if( !pt_found )
        return 0;

    ///get normal of surface_pt, and calc surface by the surface_pt and its normal
    std::vector<int> pointIndicesOut;
    std::vector<float> pointRadiusSquaredDistance;
    kdtree_.nearestKSearch( surface_pt, 4*r_nearest*r_nearest, pointIndicesOut, pointRadiusSquaredDistance );
    uint id_nn = 0;
    for(id_nn=0; id_nn<pointIndicesOut.size() && !pcl::isFinite( normals_->at(pointIndicesOut[id_nn]) ); id_nn++)
        ;
    if( id_nn == pointIndicesOut.size() )
        return 0;
    const pcl::Normal &plane_norm = normals_->at(pointIndicesOut[id_nn]);
    _plane_coef[0] = plane_norm.normal_x;
    _plane_coef[1] = plane_norm.normal_y;
    _plane_coef[2] = plane_norm.normal_z;
    _plane_coef[3] = surface_pt.x*_plane_coef[0]+surface_pt.y*_plane_coef[1]+surface_pt.z*_plane_coef[2];
    _plane_err = plane_norm.curvature;

    ///特征点三维坐标
    float pt_xn = (_pt.x-camera_cx + 0.5) * _1_camera_fx;
    float pt_yn = (_pt.y-camera_cy + 0.5) * _1_camera_fy;
    _pt3d.z = _plane_coef[3] / ( pt_xn*_plane_coef[0]+pt_yn*_plane_coef[1]+_plane_coef[2] );
    _pt3d.x = pt_xn * _pt3d.z;
    _pt3d.y = pt_yn * _pt3d.z;


    if(SHOW_TIME_INFO)
    {
        gettimeofday(&timen,NULL);
        std::cout << " norm time "  << (timen.tv_sec-timel.tv_sec)*1000+(timen.tv_usec-timel.tv_usec)/1000.0;
        timel = timen;
    }
    return point_num;
}

uint SpatialInvariantColorFeature::calcPt6dSVD(const cv::Point& _pt, cv::Point3f &_pt3d, cv::Vec4d &_plane_coef, double &_plane_err, const uint &SPATIAL_RADIUS )
{//_plan_coef: Ax+By+Cz=D; return num of pixels

    const bool SHOW_TIME_INFO = false;
    timeval time0, timel, timen;
    if(SHOW_TIME_INFO)
    {
        gettimeofday(&timel,NULL);
        time0 = timel;
    }

    const uint r_nearest = 3;//r_roi/2;//首先用一个较小的临域来估计前景的深度范围，如果直接用大ROI的话，有可能会检测到其他更前的前景
    cv::Mat nearest_roi(depth_16U_, cv::Rect(_pt.x-r_nearest, _pt.y-r_nearest, r_nearest*2+1, r_nearest*2+1));//without copy data,only new header

    ///截取特征点处的ROI
    const uint THRESH_NOISE_POINTS = 1;//如果某一个深度的点，总点数少于此，则认为是噪声
    ///构建小范围深度直方图(量化至1280级，分辨率为1cm)
    const uint HIST_RESOLUTION = 10;//unit:mm
    const uint DEPTH_MAX = 12800;//unit:mm
    std::vector<ushort> depth_8U_hist(DEPTH_MAX/HIST_RESOLUTION,0);
    uint min_depth_8U = DEPTH_MAX/HIST_RESOLUTION-1;
    uint max_depth_8U = 0;
    uchar *pdata;
    uint point_num = 0;
    uint depth_uchar;
    for(size_t i =0; i<r_nearest*2+1; i++)
    {
        pdata = nearest_roi.data + i*nearest_roi.step[0];
        for(size_t j=0; j<r_nearest*2+1; j++)
        {
            const u_int16_t &depth_current = *(u_int16_t*)pdata;
            depth_uchar = depth_current>=DEPTH_MAX ? 0 : depth_current/HIST_RESOLUTION;
            if( depth_uchar != 0 )
            {
                point_num ++;
                depth_8U_hist[depth_uchar]++;
                if( depth_uchar<min_depth_8U && depth_uchar!=0 ) min_depth_8U = depth_uchar;
                if( depth_uchar>max_depth_8U ) max_depth_8U = depth_uchar;
            }
            pdata += nearest_roi.step[1];
        }
    }
    if( point_num<THRESH_NOISE_POINTS*2 )
        return 0;

    ///求取小范围前景的深度范围
    for( ; depth_8U_hist[min_depth_8U]<THRESH_NOISE_POINTS && min_depth_8U!=max_depth_8U; min_depth_8U++) ;
    for( ; depth_8U_hist[max_depth_8U]<THRESH_NOISE_POINTS && min_depth_8U!=max_depth_8U; max_depth_8U--) ;
    uint front_depth_8U_thresh = min_depth_8U;
    uint gap_width = 0;
    for( size_t i=min_depth_8U+1; i<=max_depth_8U; i++ )
    {
        if( depth_8U_hist[i]<THRESH_NOISE_POINTS )
            gap_width ++;
        else if(gap_width*HIST_RESOLUTION<SPATIAL_R)
            front_depth_8U_thresh = i;
    }
    float front_thresh_max = (float)front_depth_8U_thresh*HIST_RESOLUTION+HIST_RESOLUTION-1;
    float front_thresh_min = (float)min_depth_8U*HIST_RESOLUTION;
    front_thresh_max += SPATIAL_R/2;
    front_thresh_min -= SPATIAL_R/2;

    ///surch for nearest point as the depth of the feature
    pcl::PointXYZRGB nearest_pt;
    bool pt_found = false;
    const pcl::PointXYZRGB &pt = cloud_->at(_pt.x, _pt.y);
    if( pt.z!=0 && pt.z<front_thresh_max )
    {
        pt_found = true;
        _pt3d.x = nearest_pt.x = pt.x;
        _pt3d.y = nearest_pt.y = pt.y;
        _pt3d.z = nearest_pt.z = pt.z;
    }
    for(int r=1; r<=r_nearest && !pt_found; r++)//distance
    for(int side=0; side<4    && !pt_found; side++)//four sides: clockwise
    for(int i=-r; i<r         && !pt_found; i++)
    {
        int x_off, y_off;
        if     (side==0) x_off = i, y_off=-r;//up       //  0 0 1
        else if(side==1) x_off = r, y_off= i;//right    //  3 * 1
        else if(side==2) x_off =-i, y_off= r;//down     //  3 2 2
        else if(side==3) x_off =-r, y_off=-i;//left
        const pcl::PointXYZRGB &pt = cloud_->at(_pt.x+x_off, _pt.y+y_off);
        if( pt.z!=0 && pt.z<front_thresh_max )
        {
            pt_found = true;
            nearest_pt.x = pt.x;
            nearest_pt.y = pt.y;
            nearest_pt.z = pt.z;
            float pt_xn = (_pt.x-camera_cx + 0.5) * _1_camera_fx;
            float pt_yn = (_pt.y-camera_cy + 0.5) * _1_camera_fy;
            _pt3d.z = nearest_pt.z;
            _pt3d.x = pt_xn * _pt3d.z;
            _pt3d.y = pt_yn * _pt3d.z;
        }
    }
    if( !pt_found )
        return 0;
    /// plane
    pcl::PointXYZRGB center3d;
    center3d.x = _pt3d.x, center3d.y = _pt3d.y, center3d.z = _pt3d.z;
    pcl::PointIndicesPtr inliers_roi = (pcl::PointIndicesPtr) new pcl::PointIndices ;
    std::vector<float> pointRadiusSquaredDistance;
    kdtree_.radiusSearch( center3d, SPATIAL_RADIUS, inliers_roi->indices, pointRadiusSquaredDistance );

    std::vector<int> & ids = inliers_roi->indices;
    cv::Mat plane_points = cv::Mat( ids.size(), 3, CV_64F);//所有前景点,预留足够空间，之后会截取ROI来使用
    double front_centroid_x = 0;//所有前景点的均值，用于求SVD之前先无偏
    double front_centroid_y = 0;
    double front_centroid_z = 0;
    uchar *p_plane_points = plane_points.data;
    for(int i=0; i<ids.size(); i++ )
    {
        pcl::PointXYZRGB &cur_point  = cloud_->at( ids[i] );
        ((double*)p_plane_points)[0] = cur_point.x;
        ((double*)p_plane_points)[1] = cur_point.y;
        ((double*)p_plane_points)[2] = cur_point.z;
        front_centroid_x += cur_point.x;
        front_centroid_y += cur_point.y;
        front_centroid_z += cur_point.z;
        p_plane_points  += plane_points.step[0];
    }
    front_centroid_x /= ids.size();
    front_centroid_y /= ids.size();
    front_centroid_z /= ids.size();
    //正常的SVD求取平面的方法，需要每个点都减去中心均值点，相当于无偏的过程。这本质上是要求将平面过原点。
    p_plane_points = plane_points.data;
    for( size_t i=0; i<ids.size(); i++)
    {
        ((double*)p_plane_points)[0] -= front_centroid_x;
        ((double*)p_plane_points)[1] -= front_centroid_y;
        ((double*)p_plane_points)[2] -= front_centroid_z;
        p_plane_points += plane_points.step[0];
    }
    cv::Mat U, S, VT; //A=U*S*VT; S为特征值，VT为特征向量。opencv中解得的是V的转置，matlab得到的直接是V
    cv::SVDecomp(plane_points,S,U,VT);//,cv::SVD::FULL_UV);  //后面的FULL_UV表示把U和VT补充成单位正交方阵;
//    std::cout << "S="  << std::endl << S  << std::endl;
//    std::cout << "VT=" << std::endl << VT << std::endl;
    //最小的特征向量即为平面的法向向量，即平面上点与之正交: ([x,y,z]-front_centroid)*VT(3)=0
    double * axis_x = (double*)(VT.data+VT.step[0]*0);
    double * axis_y = (double*)(VT.data+VT.step[0]*1);
    double * axis_z = (double*)(VT.data+VT.step[0]*2);
    std::memcpy( _plane_coef.val, axis_z, sizeof(double)*3 );
    _plane_coef[3] = front_centroid_x*axis_z[0]+front_centroid_y*axis_z[1]+front_centroid_z*axis_z[2];
    _plane_err = ( (double*)S.data)[2] / hypot( ((double*)S.data)[0], ((double*)S.data)[1] );

    //ransac plane fitting
//    pcl::SACSegmentation<pcl::PointXYZRGB> segmenter_plane;
//    pcl::PointIndices inliers_plane;
//    pcl::ModelCoefficients coefficients_plane;
//    segmenter_plane.setOptimizeCoefficients (true);
//    segmenter_plane.setModelType (pcl::SACMODEL_PLANE);
//    segmenter_plane.setMethodType (pcl::SAC_RANSAC);
//    segmenter_plane.setDistanceThreshold ( 20 );
//    segmenter_plane.setInputCloud ( cloud_->makeShared() );
//    segmenter_plane.setIndices( inliers_roi );
//    segmenter_plane.segment (inliers_plane, coefficients_plane);
//    if ( inliers_plane.indices.size() == 0 )
//        return 0;
//    _plane_coef[0] = coefficients_plane.values[0];
//    _plane_coef[1] = coefficients_plane.values[1];
//    _plane_coef[2] = coefficients_plane.values[2];
//    _plane_coef[3] = coefficients_plane.values[3];
//    _plane_err = 0;

    if(SHOW_TIME_INFO)
    {
        gettimeofday(&timen,NULL);
        std::cout << " norm time "  << (timen.tv_sec-timel.tv_sec)*1000+(timen.tv_usec-timel.tv_usec)/1000.0;
        timel = timen;
    }
    return point_num;
}
cv::Mat_<bool>
SpatialInvariantColorFeature::extractImgPatch( const cv::Point3f &_pt3d, const uint &SPATIAL_RADIUS )
{
    cv::Mat_<bool> flag( width, height );
    flag.setTo( false );
    uint border[4] = { width, height, 0, 0 };
    const int thresh_x_min = _pt3d.x - SPATIAL_RADIUS;
    const int thresh_x_max = _pt3d.x + SPATIAL_RADIUS;
    const int thresh_y_min = _pt3d.y - SPATIAL_RADIUS;
    const int thresh_y_max = _pt3d.y + SPATIAL_RADIUS;
    const int thresh_z_min = _pt3d.z - SPATIAL_RADIUS;
    const int thresh_z_max = _pt3d.z + SPATIAL_RADIUS;
    const uint SPATIAL_RADIUS2 = SPATIAL_RADIUS*SPATIAL_RADIUS;
    for( int h=0; h<height; h++)
    {
        uchar *p_flag = flag.data + h*flag.step[0];
        for( int w=0; w<width; w++, p_flag += flag.step[1] )
        {
            pcl::PointXYZRGB &pt3d = cloud_->at(w,h);
            ///fast check if outside radius
            if( pt3d.x<thresh_x_min || pt3d.x>thresh_x_max
             || pt3d.y<thresh_y_min || pt3d.y>thresh_y_max
             || pt3d.z<thresh_z_min || pt3d.z>thresh_z_max)
                continue;
            const uint x = abs( pt3d.x - _pt3d.x);
            const uint y = abs( pt3d.y - _pt3d.y);
            const uint z = abs( pt3d.z - _pt3d.z);
            if( x+y+z > SPATIAL_RADIUS )
            if( x*x+y*y+z*z > SPATIAL_RADIUS2 )
                continue;
            ///inside radius
            *p_flag = true;
            if( w<border[0] )
                border[0] = w;
            else if( w>border[2] )
                border[2] = w;
            if( h<border[1] )
                border[1] = h;
            else if( h>border[3] )
                border[3] = h;
        }
    }
    return flag;
}

uint
SpatialInvariantColorFeature::warpPerspectivePatch( const cv::Point3f &_pt3d, const cv::Vec4d _plane_coef , cv::Mat &_feature_patch, const uint &SPATIAL_RADIUS)
{
    assert( _feature_patch.cols == PATCH_SIZE && _feature_patch.rows == PATCH_SIZE );
    assert( _feature_patch.channels()==4 );
    _feature_patch.setTo(0);

    const bool SHOW_TIME_INFO = false;
    timeval time0, timel, timen;
    if(SHOW_TIME_INFO)
    {
        gettimeofday(&timel,NULL);
        time0 = timel;
    }

    ///计算透视变换矩阵Calc perspective mat
    cv::Mat_<double> rotat_mat(3,3);
    cv::Vec3d axes_morm( _plane_coef[0], _plane_coef[1], _plane_coef[2] );
    cv::Vec3d axes_view( -_pt3d.x, -_pt3d.y, -_pt3d.z );
    axes_view /= cv::norm(axes_view);
    cv::Vec3d rotat_axes = axes_morm.cross( axes_view );
    double   rotat_angle = asin( cv::norm(rotat_axes) );
    rotat_axes *= rotat_angle / cv::norm(rotat_axes);
    cv::Rodrigues( rotat_axes, rotat_mat );
    cv::Mat_<int32_t> rotat_mat32S256(3,3);
    rotat_mat.convertTo( rotat_mat32S256, CV_32S, 256);

    /*
    ///假设二维图像上的所有点都在这个平面上，然后进行变换，等价于透视变换
    cv::Mat img_after = cv::Mat::zeros(PATCH_SIZE, PATCH_SIZE, CV_8UC3);
    double W2P_RATIO = double(PATCH_SIZE/2)/(double)SPATIAL_RADIUS;//空间尺度mm到图像片像素的变换比例
    double perspect_scale = _pt3d.z*m_1_camera_fx * W2P_RATIO;
    cv::Mat_<double> perspec_mat = cv::Mat_<double>::eye(3,3);
    perspec_mat(0,0) = ( rotat_mat(0,0) - rotat_mat(0,2)*axes_morm(0)/axes_morm(2) ) * perspect_scale;
    perspec_mat(0,1) = ( rotat_mat(0,1) - rotat_mat(0,2)*axes_morm(1)/axes_morm(2) ) * perspect_scale;
    perspec_mat(1,0) = ( rotat_mat(1,0) - rotat_mat(1,2)*axes_morm(0)/axes_morm(2) ) * perspect_scale;
    perspec_mat(1,1) = ( rotat_mat(1,1) - rotat_mat(1,2)*axes_morm(1)/axes_morm(2) ) * perspect_scale;
    perspec_mat(0,2) =  PATCH_SIZE/2 - ( _pt.x*perspec_mat(0,0) + _pt.y*perspec_mat(0,1) );
    perspec_mat(1,2) =  PATCH_SIZE/2 - ( _pt.x*perspec_mat(1,0) + _pt.y*perspec_mat(1,1) );
    cv::warpPerspective( rgb_img_, img_after, perspec_mat, cv::Size(PATCH_SIZE,PATCH_SIZE) );
    cv::cvtColor( img_after, _feature_patch, cv::COLOR_BGR2BGRA );
//    cv::imshow( "perspectiveTransform", _feature_patch );
//    cv::waitKey();
    int neighbor_num = PATCH_SIZE*PATCH_SIZE;
    int total_num = PATCH_SIZE*PATCH_SIZE;
/*/

    ///将前景点映射到正视平面上(x2d,y2d)
    pcl::PointXYZRGB center3d;
    center3d.x = _pt3d.x, center3d.y = _pt3d.y, center3d.z = _pt3d.z;
    std::vector<int> pointIndicesOut;
    std::vector<float> pointRadiusSquaredDistance;
    kdtree_.radiusSearch( center3d, SPATIAL_RADIUS, pointIndicesOut, pointRadiusSquaredDistance );///std::max(cos(rotat_angle),0.5)
    int neighbor_num = pointIndicesOut.size();
    if(SHOW_TIME_INFO)
    {
        gettimeofday(&timen,NULL);
        std::cout << "  search " << neighbor_num << " cost " << (timen.tv_sec-timel.tv_sec)*1000+(timen.tv_usec-timel.tv_usec)/1000.0;
        timel = timen;
    }
    int total_num = 0;
    int W2P_RATIO_256 = double(PATCH_SIZE/2)/(double)SPATIAL_RADIUS * 256;//空间尺度mm到图像片像素的变换比例
    std::vector<int> pixel_radius_table;    //某个像素在某个深度下所对应的图像片像素尺度(因此patch_map_大小有关)，用于视角变换时插值放大, 输入为 depth/100
    pixel_radius_table.resize(128);
    for(uint i=0; i<pixel_radius_table.size(); i++)
        pixel_radius_table[i] = (i*100+99)*_1_camera_fy * W2P_RATIO_256/256 *2;
    cv::Mat_<int> mask( PATCH_SIZE, PATCH_SIZE, -(int)SPATIAL_RADIUS );
    for(int i=0; i< neighbor_num; i++)
    {
        pcl::PointXYZRGB &cur_point = cloud_->at(pointIndicesOut[i]);
        int xn = cur_point.x - center3d.x + 0.5;
        int yn = cur_point.y - center3d.y + 0.5;
        int zn = cur_point.z - center3d.z + 0.5;
        int proj_x = ( xn*rotat_mat32S256(0,0) + yn*rotat_mat32S256(0,1) + zn*rotat_mat32S256(0,2) ) / 256;
        int proj_y = ( xn*rotat_mat32S256(1,0) + yn*rotat_mat32S256(1,1) + zn*rotat_mat32S256(1,2) ) / 256;
        int proj_z = ( xn*rotat_mat32S256(2,0) + yn*rotat_mat32S256(2,1) + zn*rotat_mat32S256(2,2) ) / 256;

        ///test
//        if( proj_x!=0 && proj_y!=0 )
//        {
//            double proj_ratio = sqrt( xn*xn + yn*yn + zn*zn ) / sqrt( proj_x*proj_x + proj_y*proj_y );
//            proj_x *= proj_ratio;
//            proj_y *= proj_ratio;
//        }

        ///test
//        uint32_t cur_color = proj_z *256 / (int)SPATIAL_RADIUS;
//        if     ( cur_color >  127 ) cur_color =  127;
//        else if( cur_color < -127 ) cur_color = -127;
//         cur_color = cur_color + 127;
//        cur_color = cur_color<<16 | cur_color<<8 | cur_color<<0;

        int x_pixel = proj_x * W2P_RATIO_256 /256;//计算它在正视图像中的坐标
        int y_pixel = proj_y * W2P_RATIO_256 /256;//计算它在正视图像中的坐标
        if( x_pixel*x_pixel + y_pixel*y_pixel > (PATCH_SIZE/2)*(PATCH_SIZE/2) )
            continue;
        x_pixel += PATCH_SIZE/2, y_pixel += PATCH_SIZE/2;
        uint id_r = std::min( (uint)cur_point.z/100, (uint)pixel_radius_table.size()-1 );
        int& pixel_r = pixel_radius_table[id_r];//当前像素点所占据的像素面积（由于远近不同，空间尺度也不同）
        int min_px = x_pixel-pixel_r, max_px = x_pixel+pixel_r;//为了区域增长, 再额外增大一个像素
        int min_py = y_pixel-pixel_r, max_py = y_pixel+pixel_r;//为了区域增长, 再额外增大一个像素
        for(int y=min_py; y<=max_py; y++)
        if( y>=0 && y<(int)PATCH_SIZE )
        {
            uchar *p_patch = _feature_patch.data + y*_feature_patch.step[0] + min_px*_feature_patch.step[1];
            uchar *p_mask  = mask.data + y*mask.step[0] + min_px*mask.step[1];
            for(int x=min_px; x<=max_px; x++)
            {
                if( x>=0 && x<(int)PATCH_SIZE  )
                if( proj_z>*(int*)p_mask )
                {
                    *(uint32_t*)p_patch = cur_point.rgba;
                    *(int*)p_mask = proj_z;
                }
                p_patch += _feature_patch.step[1];
                p_mask  += mask.step[1];
            }
        }
        total_num ++;
    }

//    std::cout << "valid_cubes =" << valid_cubes << "/" << cube3.size() << std::endl;

//    /// Blur and Grow is really necessary here!!!
//    cv::Mat feature_patch_blur = _feature_patch.clone();
//    static const cv::Mat kernel_d = cv::getGaussianKernel( PATCH_SIZE/10, 10, CV_64F );
//    cv::Mat kernel = cv::Mat( kernel_d.rows, kernel_d.cols, CV_8U);
//    kernel_d.convertTo( kernel, CV_8U, 255 );

//    for(int h=0; h<feature_patch_blur.rows; h++ )
//    {
//    uchar *p_data = feature_patch_blur.data + h*feature_patch_blur.step[0];
//    for(int w=0; w<feature_patch_blur.cols; w++)
//    {
//        for(int hk=0; hk<kernel.rows; hk++)
//        {
//        uchar *p_kernel =  kernel.data + hk*kernel.step[0];
//        for(int wk=0; wk<kernel.cols; wk++)
//        {
//             p_kernel += kernel.step[1];
//        }
//        }
//        p_data += feature_patch_blur.step[1];
//    }
//    }
//*/
    if(SHOW_TIME_INFO)
    {
        gettimeofday(&timen,NULL);
        std::cout << "  映射 "  << total_num << "pt cost " << (timen.tv_sec-timel.tv_sec)*1000+(timen.tv_usec-timel.tv_usec)/1000.0;
        std::cout <<"  共:" << (timen.tv_sec-time0.tv_sec)*1000+(timen.tv_usec-time0.tv_usec)/1000.0<<"ms"<<std::endl;
    }
    return neighbor_num;
}

const uint LAYERS = 3;
uint
SpatialInvariantColorFeature::sampleCubeEvenly( const cv::Point3f &_pt3d, const cv::Vec4d _plane_coef , std::vector<cv::Vec3i> & _cube, const uint &SPATIAL_RADIUS, const double &_main_angle )
{
    ///计算透视变换矩阵Calc perspective mat
    cv::Mat_<double> rotat_mat(3,3);
    cv::Vec3d axes_morm( _plane_coef[0], _plane_coef[1], _plane_coef[2] );
    cv::Vec3d axes_view( -_pt3d.x, -_pt3d.y, -_pt3d.z );
    axes_view /= cv::norm(axes_view);
    cv::Vec3d rotat_axes = axes_morm.cross( axes_view );
    double   rotat_angle = asin( cv::norm(rotat_axes) );
    rotat_axes *= rotat_angle / cv::norm(rotat_axes);
    cv::Rodrigues( rotat_axes, rotat_mat );
    if( _main_angle!=0 )
    {
        cv::Mat_<double> rotat_mat2(3,3);
        rotat_axes = cv::Vec3d(0,0,_main_angle);
        cv::Rodrigues( rotat_axes, rotat_mat2 );
        rotat_mat = rotat_mat2 * rotat_mat;
    }
    cv::Mat_<int32_t> rotat_mat32S256(3,3);
    rotat_mat.convertTo( rotat_mat32S256, CV_32S, 256);

    ///将前景点映射到正视方向的一个空间立方体中
    int BORDER_LENGTH = std::pow(3,LAYERS);
    _cube.resize( BORDER_LENGTH*BORDER_LENGTH*BORDER_LENGTH, cv::Vec3i(0,0,0) );
    std::vector<int> cube_hi_cnt( _cube.size(), 0 );

    ///将前景点映射到正视平面上(x2d,y2d)
    pcl::PointXYZRGB center3d;
    center3d.x = _pt3d.x, center3d.y = _pt3d.y, center3d.z = _pt3d.z;
    std::vector<int> pointIndicesOut;
    std::vector<float> pointRadiusSquaredDistance;
    kdtree_.radiusSearch( center3d, SPATIAL_RADIUS, pointIndicesOut, pointRadiusSquaredDistance );///std::max(cos(rotat_angle),0.5)
    int neighbor_num = pointIndicesOut.size();

    uint valid_cubes = 0;
    for(int i=0; i< neighbor_num; i++)
    {
        pcl::PointXYZRGB &cur_point = cloud_->at(pointIndicesOut[i]);
        int xn = cur_point.x - center3d.x + 0.5;
        int yn = cur_point.y - center3d.y + 0.5;
        int zn = cur_point.z - center3d.z + 0.5;
        long proj_x = ( xn*rotat_mat32S256(0,0) + yn*rotat_mat32S256(0,1) + zn*rotat_mat32S256(0,2) ) / 256;
        long proj_y = ( xn*rotat_mat32S256(1,0) + yn*rotat_mat32S256(1,1) + zn*rotat_mat32S256(1,2) ) / 256;
        long proj_z = ( xn*rotat_mat32S256(2,0) + yn*rotat_mat32S256(2,1) + zn*rotat_mat32S256(2,2) ) / 256;
        int cube_idx = (proj_x+SPATIAL_RADIUS) * BORDER_LENGTH / (SPATIAL_RADIUS*2+1);
        int cube_idy = (proj_y+SPATIAL_RADIUS) * BORDER_LENGTH / (SPATIAL_RADIUS*2+1);
        int cube_idz = (proj_z+SPATIAL_RADIUS) * BORDER_LENGTH / (SPATIAL_RADIUS*2+1);
        assert( cube_idx<BORDER_LENGTH && cube_idy<BORDER_LENGTH && cube_idz<BORDER_LENGTH );
        int cube_id = cube_idz*BORDER_LENGTH*BORDER_LENGTH + cube_idy*BORDER_LENGTH + cube_idx;
        _cube[ cube_id ] += cv::Vec3i( cur_point.r, cur_point.g, cur_point.b );
        cube_hi_cnt[ cube_id ] ++;
    }
    for(int i=0; i<_cube.size(); i++ )
        if( cube_hi_cnt.at(i) > 0 )
        {
            _cube[i] /= cube_hi_cnt[i];
            valid_cubes ++;
        }

    return valid_cubes;
}

std::vector<cv::Vec3i> SpatialInvariantColorFeature::PyramidCube( const std::vector<cv::Vec3i> & _cube_hi_res )
{
    const int BORDER_LENGTH = std::pow(3,LAYERS);
    std::vector<cv::Vec3i> _cube;
    _cube.resize( LAYERS*26 +1 );//每层是一个26临域
    uint cube_id = 0;
    _cube[ cube_id++ ] = _cube_hi_res[ _cube_hi_res.size()/2 ];
    for( int l=1; l<=LAYERS; l++ )
    {
        const uint cell_length = std::pow(3,l-1);
        for(int neighbor_x=-1; neighbor_x<=1; neighbor_x++ )
        for(int neighbor_y=-1; neighbor_y<=1; neighbor_y++ )
        for(int neighbor_z=-1; neighbor_z<=1; neighbor_z++ )
        {///26 neighbors
            if( neighbor_x==0 && neighbor_y==0 && neighbor_z==0 )
                continue;
            const uint cell_center_x = neighbor_x * cell_length + BORDER_LENGTH/2;
            const uint cell_center_y = neighbor_y * cell_length + BORDER_LENGTH/2;
            const uint cell_center_z = neighbor_z * cell_length + BORDER_LENGTH/2;
            cv::Vec3i cell_rgb(0,0,0);
            uint cell_rgb_cnt = 0;
            for( int x=cell_center_x-cell_length/2; x<=cell_center_x+cell_length/2; x++ )
            for( int y=cell_center_y-cell_length/2; y<=cell_center_y+cell_length/2; y++ )
            for( int z=cell_center_z-cell_length/2; z<=cell_center_z+cell_length/2; z++ )
            {
                if( x<0 || x>=BORDER_LENGTH || y<0 || y>=BORDER_LENGTH || z<0 || z>=BORDER_LENGTH )
                    continue;
                const cv::Vec3i &cur_rgb = _cube_hi_res[ z*BORDER_LENGTH*BORDER_LENGTH + y*BORDER_LENGTH + x ];
                if( cur_rgb != cv::Vec3i::zeros() )
                {
                    cell_rgb_cnt ++;
                    cell_rgb += cur_rgb;
                }
            }
            cell_rgb_cnt = cell_rgb_cnt == 0 ?  1 : cell_rgb_cnt;
            _cube[ cube_id++ ] = cell_rgb / (int)cell_rgb_cnt;
        }
    }
    return _cube;
}


uint SpatialInvariantColorFeature::calcFeatureDir(const cv::Mat& _feature_patch, cv::Point2d &_main_dir, const double &_dense_thresh)
{
    /// 统计区块颜色
    std::vector<int64> color_r(annular_mask_->TOTAL_CELLS,0);
    std::vector<int64> color_g(annular_mask_->TOTAL_CELLS,0);
    std::vector<int64> color_b(annular_mask_->TOTAL_CELLS,0);
    std::vector<int> color_weight(annular_mask_->TOTAL_CELLS,0);
    for(int h=0; h<_feature_patch.rows; h++)
    {
        uchar * p_cur_patch = _feature_patch.data + h*_feature_patch.step[0];
        for(int w=0; w<_feature_patch.cols; w++)
        {
            if( *(uint32_t*)p_cur_patch != 0 )
            {
                uchar pos = annular_mask_->getCellID(w,h);
                if(pos!=annular_mask_->BAD_CELL)
                {
                    color_r[pos] += p_cur_patch[2];
                    color_g[pos] += p_cur_patch[1];
                    color_b[pos] += p_cur_patch[0];
                    color_weight[pos] ++;
                }
            }
            p_cur_patch += _feature_patch.step[1];
        }
    }


    /// 计算主方向向量main_dir_vec
    std::vector<uint> dir_bright(annular_mask_->ANGLE_RES,0);//方向亮度
    cv::Point2d main_dir_vec(0,0);//主方向的方向向量
    cv::Point2d valid_dir_vec(0,0);//
    uint valid_cells_cnt = 0;
    const int dense_thresh = PATCH_SIZE*PATCH_SIZE/annular_mask_->TOTAL_CELLS * _dense_thresh;
    for(uint i=0; i<annular_mask_->TOTAL_CELLS; i++)
    {
        if( color_weight[i] > dense_thresh )
        {
            color_r[i] /= color_weight[i];
            color_g[i] /= color_weight[i];
            color_b[i] /= color_weight[i];

            int64 &r = color_r[i];
            int64 &g = color_g[i];
            int64 &b = color_b[i];
            int I = (r+g+b)/3;
            if( i!=0 )    //i==0 means the center cell
            {
//                if( i > annular_mask_->ANGLE_RES )//去掉不鲁棒的第一层
                if( i <= annular_mask_->ANGLE_RES ) //外层形状不稳定
                    dir_bright[(i-1)%annular_mask_->ANGLE_RES] +=  I;//[0~255]
                main_dir_vec += annular_mask_->DIR_PER_CELL[i] * I;
                valid_dir_vec+= annular_mask_->DIR_PER_CELL[i];
            }
            valid_cells_cnt ++;
        }
     }
    if( valid_cells_cnt <= 6 )
        return 0;

    ///valid_dir_vec方向的正负90度内全是有效点，剩下的全是无效点；则说明是直边缘（非空间角点）
    uint main_cnt = 0;
    uint else_cnt = 0;
    for(size_t i=0; i<dir_bright.size(); i++)
    {
        if(dir_bright[i])
        {
            cv::Point2d cur_dir = annular_mask_->DIR_PER_CELL[i+1];
            double cos_angle = cur_dir.dot( valid_dir_vec ) / hypot(cur_dir.x,cur_dir.y) / hypot(valid_dir_vec.x,valid_dir_vec.y);
            if( cos_angle > cos( (90-20)*M_PI/180.0 ) )
                main_cnt ++;
            else if( cos_angle < cos( (90+20)*M_PI/180.0 ) )
                else_cnt ++;
        }
    }
    if( main_cnt > annular_mask_->ANGLE_RES*0.35 && else_cnt <= 2 )
    {
        double cos_theta = valid_dir_vec.dot( main_dir_vec ) / hypot(valid_dir_vec.x,valid_dir_vec.y) / hypot(main_dir_vec.x,main_dir_vec.y);
//        if( fabs(cos_theta)>cos(M_PI/9) )//不是颜色角点
            return 0;
    }


    ///3.4 根据灰度值直方图判断是否是有效角点（灰度值突变或空间突变）
//    uint bright_max = 0, bright_min = 255*annular_mask_->DIST_RES;
    uint valid_dir_cnt = 0;
    uint max_dir_cnt = 0;
    for(size_t i=0; i<dir_bright.size()*1.5; i++)
    {
        if(dir_bright[i%dir_bright.size()])
        {
            valid_dir_cnt ++;
        }
        else
        {
            if( valid_dir_cnt > max_dir_cnt )
                max_dir_cnt = valid_dir_cnt;
            valid_dir_cnt = 0;
        }
    }
    double cos_theta = valid_dir_vec.dot( main_dir_vec ) / hypot(valid_dir_vec.x,valid_dir_vec.y) / hypot(main_dir_vec.x,main_dir_vec.y);
    int valid_angle_range = max_dir_cnt * 360 / dir_bright.size();
    if( valid_angle_range > 120 && valid_angle_range < 220 )//近似180度,不是空间角点
    {
//        if( fabs(cos_theta)>cos(M_PI/9) )//不是颜色角点
//            return 0;
    }

    _main_dir = main_dir_vec;
    return valid_cells_cnt;
}

uint
SpatialInvariantColorFeature::generateFeatureCode(const cv::Mat& _feature_patch, const cv::Point2d &_main_dir, cv::Mat& _color_code, const double &_dense_thresh)
{
    double main_dir_deg = _main_dir.x==0 ? (_main_dir.y>0?M_PI_2:3*M_PI_2) : atan( _main_dir.y / _main_dir.x );//主方向的角度值
    if( _main_dir.x<0 ) main_dir_deg += M_PI;
    if( main_dir_deg<0 ) main_dir_deg += 2*M_PI;
    main_dir_deg = main_dir_deg*180.0/M_PI;
    /// 统计区块颜色
    std::vector<uint64>    color_r(patch_mask_->TOTAL_CELLS,0);
    std::vector<uint64>    color_g(patch_mask_->TOTAL_CELLS,0);
    std::vector<uint64>    color_b(patch_mask_->TOTAL_CELLS,0);
    std::vector<uint> color_weight(patch_mask_->TOTAL_CELLS,0);
    for(int h=0; h<_feature_patch.rows; h++)
    {
        uchar * p_cur_patch = _feature_patch.data + h*_feature_patch.step[0];
        for(int w=0; w<_feature_patch.cols; w++)
        {
            if( *(uint32_t*)p_cur_patch != 0 )
            {
                uchar cell_id = patch_mask_->getCellID( w, h, main_dir_deg );
                if( cell_id != patch_mask_->BAD_CELL )
                {
                    color_r[cell_id] += p_cur_patch[2];
                    color_g[cell_id] += p_cur_patch[1];
                    color_b[cell_id] += p_cur_patch[0];
                    color_weight[cell_id] ++;
                }
            }
            p_cur_patch += _feature_patch.step[1];
        }
    }
    /// 对颜色进行编码
    if( _color_code.cols!=patch_mask_->TOTAL_CELLS || _color_code.type()!=color_encoder_.code_type_ )
        _color_code.create( 1, patch_mask_->TOTAL_CELLS, color_encoder_.code_type_ );
    uint mean_V = 0, pixel_cnt = 0;
    for(uint i=0; i<patch_mask_->TOTAL_CELLS; i++)
    {
        mean_V += ( color_r[i] +  color_g[i] +  color_b[i] ) / 3;
        pixel_cnt += color_weight[i];
    }
    if( pixel_cnt==0 ) return 0;
    mean_V /= pixel_cnt;
    uint valid_cells_cnt = 0;
    const uint dense_thresh = PATCH_SIZE*PATCH_SIZE/patch_mask_->TOTAL_CELLS * _dense_thresh;
    uchar *p_code = _color_code.data;
    for(uint i=0; i<patch_mask_->TOTAL_CELLS; i++)
    {
        if( color_weight[i] > dense_thresh )
        {
            color_r[i] /= color_weight[i];
            color_g[i] /= color_weight[i];
            color_b[i] /= color_weight[i];
            uint64 &r = color_r[i];
            uint64 &g = color_g[i];
            uint64 &b = color_b[i];
            color_encoder_.encode( p_code, r, g, b );
            valid_cells_cnt ++;
        }
        else
            color_encoder_.invalidCode( p_code );
        p_code += _color_code.step[1];
     }
    return valid_cells_cnt;
}

uint SpatialInvariantColorFeature::generateFeatureCode_hov(const cv::Mat& _feature_patch, cv::Mat& _color_code, const uchar& _method)
{
    if( _color_code.cols != patch_mask_->TOTAL_CELLS || _color_code.type() != CV_8UC1 )
        _color_code.create( 1, patch_mask_->TOTAL_CELLS, CV_8UC1 );//存储RGB颜色,先累加后均值

    std::vector<uint> color_hist(64,0);
    for(int h=0; h<_feature_patch.rows; h++)
    {
        uchar * p_cur_patch = _feature_patch.data + h*_feature_patch.step[0];
        for(int w=0; w<_feature_patch.cols; w++)
        {
            if( h*h+w*w < _feature_patch.rows*_feature_patch.cols )
            if( *(uint32_t*)p_cur_patch != 0 )
            {
                switch (_method)
                {
                case 0:
                default:
                    uchar id = color_encoder_.rgb2IntCode( p_cur_patch[0], p_cur_patch[1], p_cur_patch[2], 6 );
//                    std::cout<<"id="<<(int)id<<std::endl;
//                    id = ( (int)p_cur_patch[0] + p_cur_patch[1] + p_cur_patch[2] ) / 3 * 64 / 256;
                    color_hist[id] ++;
                    break;
                }
            }
            p_cur_patch += _feature_patch.step[1];
        }
    }
    for(int i=0; i<64; i++)//nomorlize the histogram
        _color_code.at<uchar>(0,i) = color_hist[i] * 255 / (_feature_patch.rows*_feature_patch.cols);
    return 64;
}
bool
SpatialInvariantColorFeature::prepareFrame( const cv::Mat _rgb_image, const cv::Mat _depth_16U)
{
    const bool SHOW_TIME_INFO = false;
    timeval time_start, time_temp;
    gettimeofday(&time_start,NULL);

    assert( _rgb_image.type()==CV_8UC1 || _rgb_image.type()==CV_8UC3 );
    assert( _depth_16U.channels()==1 );
    if( _depth_16U.type()!=CV_16UC1 ) _depth_16U.convertTo(depth_16U_,CV_16U);
    else                              _depth_16U.copyTo(   depth_16U_ );
    height = _depth_16U.rows;
    width  = _depth_16U.cols;
    assert( width>0 && height>0 );
    assert( width==_rgb_image.cols && height==_rgb_image.rows );
    if( _rgb_image.channels()==1 )
        cv::cvtColor(_rgb_image,rgb_img_,CV_GRAY2RGB);
    else if( _rgb_image.channels()==3 )
        rgb_img_ = _rgb_image.clone();
    rgb_show_ = rgb_img_.clone();

    // 生成点云
    if(!cloud_)
    {
      cloud_ = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>(width, height));
      cloud_->is_dense = false;
    }
    else if( cloud_->width != width || cloud_->height != height )
    {
        cloud_->resize( width*height );
        cloud_->width = width;
        cloud_->height= height;
    }
#pragma omp parallel for
    for( int y = 0; y < (int)height; ++y )
    {
        uchar *prgb   = _rgb_image.data + y*_rgb_image.step[0];
        uchar *pdepth = depth_16U_.data + y*depth_16U_.step[0];
        pcl::PointXYZRGB *p_cloud = &cloud_->points[0] + y*width;
        for(int x = 0; x < (int)width; x++ )
        {
            const u_int16_t &depth_current = *(u_int16_t*)pdepth;// / 1000.0f;
            if( std::isnan(depth_current) || depth_current<10 )
            {
                p_cloud->x = p_cloud->y = p_cloud->z = std::numeric_limits<float>::quiet_NaN();
            }
            else
            {
                p_cloud->z = depth_current;
                p_cloud->x = (x-camera_cx+0.5) * _1_camera_fx * depth_current;
                p_cloud->y = (y-camera_cy+0.5) * _1_camera_fy * depth_current;
                p_cloud->b = prgb[0];
                p_cloud->g = prgb[1];
                p_cloud->r = prgb[2];
            }
            prgb   += _rgb_image.step[1];
            pdepth += depth_16U_.step[1];
            p_cloud++;
        }
    }
    if(!normal_est_)
    {
        normals_ = (pcl::PointCloud<pcl::Normal>::Ptr) new pcl::PointCloud<pcl::Normal>;
        normal_est_ = (pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB,pcl::Normal>::Ptr) new pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB,pcl::Normal>;
        normal_est_->setNormalEstimationMethod( normal_est_->AVERAGE_3D_GRADIENT);
        normal_est_->setMaxDepthChangeFactor( SPATIAL_R );//threshold for computing object borders
        normal_est_->setNormalSmoothingSize( 50 );//估计一个特征点的法向量时所用的临域尺度;不知为何，这里用的单位是像素
    }
    normal_est_->setInputCloud( cloud_ );
    normal_est_->compute( *normals_ );
    kdtree_.setInputCloud (cloud_);

    gettimeofday(&time_temp,NULL);
    int init_time = (time_temp.tv_sec-time_start.tv_sec)*1000+(time_temp.tv_usec-time_start.tv_usec)/1000;
    if( SHOW_TIME_INFO )
        std::cout << "prepare time(ms):" << init_time << std::endl;

    return true;
}

cv::Mat
SpatialInvariantColorFeature::process( std::vector<cv::KeyPoint> &m_keypoints )
{
    const bool SHOW_TIME_INFO = false;
    timeval time_start;
    gettimeofday(&time_start,NULL);

    // 开始提取
    keypoints_filtered_.clear();
    keypoints_3D_.clear();
    features_show_.setTo(0);
    features_restore_.setTo(0);

    uint keypoint_valid_cnt =0;
    uint fake_point_cnt[3] = {0};
//#pragma omp parallel for
    for( int key_id = 0; key_id < (int)m_keypoints.size(); key_id++)
    {
        cv::Point2f keypoint2d = m_keypoints[key_id].pt;

        ///1. 提取前景cur_patch
        cv::Point3f keypoint3d;
        cv::Mat cur_patch;
        cv::Vec4d plan_coef(0,0,0,0);//Ax+By+Cz=D ,(A B C)为平面法向的单位向量
        double plane_err;
        cur_patch = cv::Mat::zeros( PATCH_SIZE, PATCH_SIZE, CV_8UC4 );

        uint front_pt_num;
//#pragma omp critical
        front_pt_num = calcPt6d( keypoint2d, keypoint3d, plan_coef, plane_err );
        if( !front_pt_num  )
        {
            fake_point_cnt[0] ++;
//            std::cout<< "\t\t\t\t\tPoint loss data."<< std::endl;
            if( DRAW_IMAG ) cv::circle(rgb_show_,  keypoint2d,2,CV_RGB(255,0,0),CV_FILLED );
            continue;
        }

        uint spatial_r = SPATIAL_R;
//        float &r = m_keypoints[key_id].size;
//        if( r != 0 )
//            spatial_r = r * _1_camera_fx * keypoint3d.z;
        warpPerspectivePatch( keypoint3d, plan_coef, cur_patch, spatial_r );
        if( 0)//fabs(plan_coef[2]) < sin(M_PI/20) && keypoint3d.z>1000 )//此处平面过于陡峭，说明特征点不鲁棒
        {
            fake_point_cnt[1] ++;
//            std::cout<< "\t\t\t\t\tPoint at curve."<< std::endl;
            if( DRAW_IMAG ) cv::circle(rgb_show_,  keypoint2d,2,CV_RGB(255,255,0),CV_FILLED );
            continue;
        }

        ///2. 计算cur_patch主方向
        cv::Point2d main_dir_vec(1,0);
        if( !calcFeatureDir( cur_patch, main_dir_vec ) )
        {
//            if( DRAW_IMAG ) cv::circle(rgb_show_,  keypoint2d,2,CV_RGB(100,100,255),CV_FILLED );
//            cv::imshow( "img", rgb_show_ );
//            cv::imshow( "fake point", cur_patch );
//            cv::waitKey();
            if( DRAW_IMAG ) cv::circle(rgb_show_,  keypoint2d,2,CV_RGB(255,0,255),CV_FILLED );
            fake_point_cnt[2] ++;
            continue;
        }
        double main_dir_rad = main_dir_vec.x==0 ? (main_dir_vec.y>0?M_PI_2:3*M_PI_2) : atan( main_dir_vec.y / main_dir_vec.x );//主方向的角度值
        if( main_dir_vec.x<0 ) main_dir_rad += M_PI;
        if( main_dir_rad<0 ) main_dir_rad += 2*M_PI;
        double main_dir_deg = main_dir_rad*180.0/M_PI;

        std::vector<cv::Vec3i> cube3;//将特征点的空间临域均匀采样/插值到一个空间立方提中
        if( patch_type_ == D_TYPE_CUBE3 )
        {
            sampleCubeEvenly( keypoint3d, plan_coef, cube3, spatial_r, 0 );
            std::cout << "cube size=" << cube3.size();
            cube3 = PyramidCube( cube3 );
            std::cout << " ->" << cube3.size() << std::endl;
        }
        ///3. 从cur_patch中获得描述子
        cv::Mat cur_descriptor;
        switch (patch_type_)
        {
        case D_TYPE_ANNULAR:
        case D_TYPE_BEEHIVE:
        {
            /// 颜色编码
            const uint valid_cells_cnt = generateFeatureCode( cur_patch, main_dir_vec, cur_descriptor );
            if( valid_cells_cnt < patch_mask_->TOTAL_CELLS/16.0 )
            {
//                std::cout<< "\t\t\t\t\tSmall Patch."<< std::endl;
                if( DRAW_IMAG ) cv::circle( rgb_show_, keypoint2d, 2, CV_RGB(0,0,255), CV_FILLED );
                continue;
            }
        }
            break;
        case D_TYPE_BRIEF:
        {
            cv::BriefDescriptorExtractor brief_extractor(32);
            const uint PATCH_SIZEE = 63;
            cv::Mat image_temp;
            cv::resize(cur_patch,image_temp,cv::Size(PATCH_SIZEE,PATCH_SIZEE),0,0,cv::INTER_NEAREST);
            cv::vector<cv::KeyPoint> key_point_temp( 1, m_keypoints[key_id] );
            key_point_temp[0].pt.x = image_temp.cols/2;
            key_point_temp[0].pt.y = image_temp.rows/2;
            key_point_temp[0].size = PATCH_SIZEE;
            key_point_temp[0].angle= main_dir_deg;
            brief_extractor.compute(image_temp, key_point_temp, cur_descriptor);
            if( !cur_descriptor.rows )////特征提取失败
                continue;
        }
            break;
        case D_TYPE_ORB:
        {
            cv::OrbDescriptorExtractor orb_extractor;
            const uint PATCH_SIZEE = 63;
            cv::Mat image_temp;
            cv::resize(cur_patch,image_temp,cv::Size(PATCH_SIZEE,PATCH_SIZEE),0,0,cv::INTER_NEAREST);
            cv::vector<cv::KeyPoint> key_point_temp( 1, m_keypoints[key_id] );
            key_point_temp[0].pt.x = image_temp.cols/2;
            key_point_temp[0].pt.y = image_temp.rows/2;
            key_point_temp[0].size = PATCH_SIZEE;
            key_point_temp[0].angle= main_dir_deg;
            orb_extractor( image_temp, cv::Mat::ones(image_temp.rows,image_temp.cols,CV_8UC1), key_point_temp, cur_descriptor, true);
            if( !cur_descriptor.rows )//特征提取失败
                continue;
        }
            break;
        case D_TYPE_BRISK:
        {
            cv::BRISK brisk_extractor;
            const uint PATCH_SIZEE = PATCH_SIZE;
            cv::Mat image_temp;
            cv::resize(cur_patch,image_temp,cv::Size(PATCH_SIZEE,PATCH_SIZEE),0,0,cv::INTER_NEAREST);
            cv::vector<cv::KeyPoint> key_point_temp( 1, m_keypoints[key_id] );
            key_point_temp[0].pt.x = image_temp.cols/2;
            key_point_temp[0].pt.y = image_temp.rows/2;
            key_point_temp[0].size = PATCH_SIZEE;
            key_point_temp[0].angle= main_dir_deg;
            brisk_extractor( image_temp, cv::Mat::ones(image_temp.rows,image_temp.cols,CV_8UC1), key_point_temp, cur_descriptor, true);
            if( !cur_descriptor.rows )//特征提取失败
                continue;
        }
            break;
        case D_TYPE_SURF:
        {
            cv::SurfDescriptorExtractor surf_extractor;
            const uint PATCH_SIZEE = 63;
            cv::Mat image_temp;
            cv::resize(cur_patch,image_temp,cv::Size(PATCH_SIZEE,PATCH_SIZEE),0,0,cv::INTER_NEAREST);
            cv::vector<cv::KeyPoint> key_point_temp( 1, m_keypoints[key_id] );
            key_point_temp[0].pt.x = image_temp.cols/2;
            key_point_temp[0].pt.y = image_temp.rows/2;
            key_point_temp[0].size = PATCH_SIZEE;
            key_point_temp[0].angle= main_dir_deg;
            surf_extractor( image_temp, cv::Mat::ones(image_temp.rows,image_temp.cols,CV_8UC1), key_point_temp, cur_descriptor, true);
            if( !cur_descriptor.rows )//特征提取失败
                continue;
        }
            break;
        case D_TYPE_HISTOGRAM:
            generateFeatureCode_hov( cur_patch, cur_descriptor );
            break;
        case D_TYPE_CUBE3:
        {
            assert( cube3.size()>0 );
            cur_descriptor.create( 1, cube3.size(), color_encoder_.code_type_ );
            uchar *p_code = cur_descriptor.data;
            for(uint i=0; i<cube3.size(); i++)
            {
                int &r = cube3.at(i)(0);
                int &g = cube3.at(i)(1);
                int &b = cube3.at(i)(2);
                if( r>0 && g>0 && b>0 )
                {
                    color_encoder_.encode( p_code, r, g, b );
                }
                else
                    color_encoder_.invalidCode( p_code );
                p_code += cur_descriptor.step[1];
             }
        }
            break;
        default:
        {
        }
            break;
        }
        ///3 保存该描述子
        #pragma omp critical
        {/// 只处理MAX_KEYPOINTS个点，多余的舍弃掉
        if( keypoint_valid_cnt < MAX_KEYPOINTS )
        {
            if( descriptors_.rows  != MAX_KEYPOINTS
             || descriptors_.cols  != cur_descriptor.cols
             || descriptors_.type()!= cur_descriptor.type() )
                descriptors_.create( MAX_KEYPOINTS, cur_descriptor.cols, cur_descriptor.type() );

            memcpy( descriptors_.data+descriptors_.step[0]*keypoint_valid_cnt, cur_descriptor.data, cur_descriptor.step[0] );
            keypoints_filtered_.push_back( m_keypoints[key_id] );
            keypoints_3D_.push_back( keypoint3d );

            if( DRAW_IMAG )
            {
                const uint x_max = features_show_.cols / PATCH_SIZE;
                const uint y_max = features_show_.rows / PATCH_SIZE;
                if( keypoint_valid_cnt < x_max*y_max )
                {
                    cv::line( cur_patch, cv::Point(PATCH_SIZE/2,PATCH_SIZE/2), cv::Point(PATCH_SIZE/2+main_dir_vec.x,PATCH_SIZE/2-main_dir_vec.y),CV_RGB(255,0,0) );
                    cv::Rect draw_mask = cv::Rect( (keypoint_valid_cnt%x_max)*PATCH_SIZE,
                                                   (keypoint_valid_cnt/y_max)*PATCH_SIZE,
                                                    PATCH_SIZE, PATCH_SIZE );
                    cur_patch.copyTo( cv::Mat(features_show_,draw_mask) );
                }
            }
            keypoint_valid_cnt++;
        }
        else
            break;
        }

        if( DRAW_IMAG )
            cv::circle(rgb_show_,  keypoint2d,2,CV_RGB(0,255,0),CV_FILLED );

    }//end of for(it_keypoints=m_keypoints.begin(); it_keypoints!=m_keypoints.end(); it_keypoints++)



    if(SHOW_TIME_INFO)
    {
        std::cout <<"KeyPointNum:"<<m_keypoints.size()<<"-"<<fake_point_cnt[0]<<"-"<<fake_point_cnt[1]<<"-"<<fake_point_cnt[2]<<"="<<keypoints_filtered_.size();
        timeval time_end;
        gettimeofday(&time_end,NULL);
        int total_time = (time_end.tv_sec-time_start.tv_sec)*1000+(time_end.tv_usec-time_start.tv_usec)/1000;
        std::cout << "总耗时(ms):" << total_time << "="<< m_keypoints.size() << "*" << (double)(total_time)/m_keypoints.size() << std::endl<< std::endl;
    }
    m_keypoints = keypoints_filtered_;
    return cv::Mat( descriptors_, cv::Rect(0, 0, descriptors_.cols, m_keypoints.size()) ).clone();
}

cv::Mat
SpatialInvariantColorFeature::processFPFH(std::vector<cv::KeyPoint> &m_keypoints, const uint &SPATIAL_RADIUS )
{
    timeval time_start, time_temp;
    gettimeofday(&time_start,NULL);

    // 开始提取
    keypoints_filtered_.clear();
    keypoints_3D_.clear();
    features_show_.setTo(0);
    features_restore_.setTo(0);

    gettimeofday(&time_temp,NULL);
    int init_time = (time_temp.tv_sec-time_start.tv_sec)*1000+(time_temp.tv_usec-time_start.tv_usec)/1000;


    ///fpfh
    pcl::FPFHEstimation<pcl::PointXYZRGB,pcl::Normal,pcl::FPFHSignature33> fpfh_est;
    fpfh_est.setSearchSurface( cloud_->makeShared() );
    fpfh_est.setInputNormals( normals_->makeShared() );
    fpfh_est.setRadiusSearch( SPATIAL_RADIUS );

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr key_points( new pcl::PointCloud<pcl::PointXYZRGB>(1,m_keypoints.size()) );
    int valid_cnt = 0;
    for( int key_id = 0; key_id < (int)m_keypoints.size(); key_id++)
    {
        cv::Point2f keypoint = m_keypoints[key_id].pt;
        const uint &border = 50;
        if( keypoint.x<border || keypoint.y<border || keypoint.x>width-1-border || keypoint.y>height-1-border )
            continue;
        cv::Point3f keypoint_3d;
        cv::Vec4d plan_coef(0,0,0,0);//Ax+By+Cz=D ,(A B C)为平面法向的单位向量
        double plane_err;
        uint front_pt_num = calcPt6d( keypoint, keypoint_3d, plan_coef, plane_err );
        if( front_pt_num>0 )
        {
            key_points->at(valid_cnt).x = keypoint_3d.x;
            key_points->at(valid_cnt).y = keypoint_3d.y;
            key_points->at(valid_cnt).z = keypoint_3d.z;
            keypoints_filtered_.push_back( m_keypoints[key_id] );
            valid_cnt ++;
        }
    }
    key_points->resize( valid_cnt );
    m_keypoints = keypoints_filtered_;

    fpfh_est.setInputCloud( key_points->makeShared() );
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_signature(new pcl::PointCloud<pcl::FPFHSignature33>() );
    fpfh_est.compute( *fpfh_signature );

    if( descriptors_.rows  != MAX_KEYPOINTS
     || descriptors_.cols  != 33
     || descriptors_.type()!= CV_32F )
        descriptors_.create( MAX_KEYPOINTS, 33, CV_32F );
    for( int i = 0; i < valid_cnt; i++)
        memcpy( descriptors_.data+descriptors_.step[0]*i, fpfh_signature->at(i).histogram, sizeof(pcl::FPFHSignature33) );

    timeval time_end;
    gettimeofday(&time_end,NULL);
    int total_time = (time_end.tv_sec-time_start.tv_sec)*1000+(time_end.tv_usec-time_start.tv_usec)/1000;
    std::cout << "FPFH总耗时(ms):" << total_time << "="<< init_time << "+" << m_keypoints.size() << "*" << (double)(total_time-init_time)/m_keypoints.size() << std::endl<< std::endl;

    return cv::Mat( descriptors_, cv::Rect(0, 0, descriptors_.cols, valid_cnt) );
}

void SpatialInvariantColorFeature::match( const cv::Mat& queryDescriptors, const cv::Mat& trainDescriptors, std::vector<cv::DMatch>& matches ) const
{
    matches.reserve( queryDescriptors.rows );
    uchar *p_from = queryDescriptors.data;
    for(int id_from=0; id_from<queryDescriptors.rows; id_from++ )
    {
        int min_dist = INFINITY;
        int min_id = -1;
        uchar *p_to =  trainDescriptors.data;
        for(int id_to=0; id_to<trainDescriptors.rows; id_to++ )
        {
            uint temp_dist = color_encoder_.machCode( p_from, p_to, trainDescriptors.cols );
            if( temp_dist < min_dist )
            {
                min_dist = temp_dist;
                min_id = id_to;
            }
            p_to += trainDescriptors.step[0];
        }
        if( min_id != -1 )
            matches.push_back( cv::DMatch( id_from, min_id, min_dist) );
        p_from += queryDescriptors.step[0];
    }
    ///cross check
    for( std::vector<cv::DMatch>::iterator p_match = matches.begin(); p_match != matches.end();  )
    {
        uchar *p_to =  trainDescriptors.data + p_match->trainIdx * trainDescriptors.step[0];
        uint min_dist = p_match->distance;
        bool reject = false;
        uchar *p_from =  queryDescriptors.data;
        for(int id_from=0; id_from<queryDescriptors.rows; id_from++ )
        {
            if( id_from != p_match->queryIdx )
            if( min_dist >= color_encoder_.machCode( p_to, p_from, queryDescriptors.cols ) )
            {
                reject = true;
                break;
            }
            p_from += queryDescriptors.step[0];
        }
        if( reject )
            p_match = matches.erase( p_match );
        else
            p_match ++;
    }
}

bool
SpatialInvariantColorFeature::restore_descriptor(const cv::Mat& _descriptor)
{
    const uint& size = PATCH_SIZE;
    std::vector<uint32_t> cur_color_show;
    features_restore_.setTo(0);
    for(uint cnt=0; cnt<(uint)_descriptor.rows; cnt++)
    {
        cur_color_show.resize(patch_mask_->TOTAL_CELLS,0);
        const uint x_max = features_restore_.cols / size;
        const uint y_max = features_restore_.rows / size;
        cv::Rect cur_patch_mask = cv::Rect( (cnt%x_max)*size, (cnt/y_max)*size, size, size );
        cv::Mat cur_patch;
        if( cnt < x_max*y_max )  cur_patch = cv::Mat(features_restore_, cur_patch_mask );
        else            return true;//cur_patch = cv::Mat::zeros(height, width, CV_8UC4);

        if( patch_type_ == D_TYPE_BEEHIVE || patch_type_ == D_TYPE_ANNULAR )
        {
            //重构每个描述子的所有对应颜色cur_color_show
            uchar *p_color_code = _descriptor.data + cnt*_descriptor.step[0];
            for(int i=0; i<_descriptor.cols; i++)
            {
                cur_color_show[i] =  color_encoder_.decode( p_color_code );
                p_color_code += _descriptor.step[1];
            }
            //将对应的颜色映射到图像上
            for(uint y=0; y<size; y++)
            {
                uchar* pdata_patch =   cur_patch.data + y*  cur_patch.step[0];
                for(uint x=0; x<size; x++)
                {
                    uchar cell_id = patch_mask_->getCellID(x,y);
                    if( cell_id != patch_mask_->BAD_CELL )
                        *(uint32_t*)pdata_patch = cur_color_show[cell_id];
                    else
                        *(uint32_t*)pdata_patch = 0;
                    pdata_patch +=   cur_patch.step[1];
                }
            }
        }
        else if( patch_type_ == D_TYPE_CUBE3 )
        {
            const int CUBE_RES = std::pow( _descriptor.cols, 1.0/3) + 0.5;
            if( _descriptor.cols != CUBE_RES*CUBE_RES*CUBE_RES )
                return false;
            cv::Mat cube_show = cv::Mat::zeros( CUBE_RES, CUBE_RES, CV_8UC4 );
            uchar *p_color_code = _descriptor.data + cnt*_descriptor.step[0];
            for(int z=0; z<CUBE_RES; z++)
            for(int y=0; y<CUBE_RES; y++)
            {
                uchar *p_img = cube_show.data + y*cube_show.step[0];
                for(int x=0; x<CUBE_RES; x++)
                {
                    uint32_t rgb = color_encoder_.decode( p_color_code );
                    if(   rgb != 0 && *(uint32_t*)p_img==0 )
                        *(uint32_t*)p_img = rgb;
                    p_img += cube_show.step[1];
                    p_color_code += _descriptor.step[1];
                }
            }
            cur_patch = cv::Mat( cur_patch, cv::Rect(1,1,cur_patch.cols-2,cur_patch.rows-2) );//make a blank border
            cv::resize( cube_show, cur_patch, cv::Size(cur_patch.cols,cur_patch.rows), 0, 0, cv::INTER_NEAREST );
        }
    }
    return true;
}
