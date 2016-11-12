#ifndef COLOR_CODED_DEPTH_FEATURE_H
#define COLOR_CODED_DEPTH_FEATURE_H
//PCL includes
#include <pcl/common/time.h>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/io/pcd_io.h>

//OpenCV includes
#include <opencv2/opencv.hpp>
#include "opencv2/nonfree/nonfree.hpp"
#include <string>
#include <sstream>

///////////颜色编码相关////////////////////////////
class ColorCoding
{
public:
    enum METHOD
    {
        HAMMING_HSV422 = 0,
        HAMMING_HSV655,
        HAMMING_GRAY8,
        HAMMING_GRAY16
    };
     class CodeBase
    {
    public:
        uint32_t rgb2code_[16][16][16];    //量化RGB值为4bit 预留32bit作为code，实际可能不用这么长
        std::vector<uint32_t> code2rgba_; //输入颜色编码值，获得该颜色rgb值
        uint32_t INVALID_CODE;
        template<typename code_t> bool initByHSV( const std::vector<code_t> &H_code, const std::vector<code_t> &S_code, const std::vector<code_t> &V_code, const code_t &EMPITY_H );
        CodeBase(){}
        virtual ~CodeBase(){}
    };
    class HSV422Code : public CodeBase
    {
    public:
        typedef uchar code_t;
        HSV422Code();
    };
    class HSV655Code : public CodeBase
    {
    public:
        typedef u_int16_t code_t;
        HSV655Code();
    };
    boost::shared_ptr<CodeBase> coder_;
public:
    const METHOD method_;
    int code_type_;//CV_8U, CV_16U, etc.
    ColorCoding( const METHOD &_method=HAMMING_HSV655 );
    int encode( void* p_code, const uchar _r, const uchar _g, const uchar _b) const;///return num of bytes in p_code
    int invalidCode( void* p_code ) const;
    uchar rgb2IntCode(const uchar _r, const uchar _g, const uchar _b, const uchar _bit_length=8) const;
    uint machCode(void* _code1, void* _code2, const uint _cells=1) const;
    uint32_t decode(const void * _code) const;//return rgba formate color
};
//////////////////////////////////////////////////
/// \brief The ColorCodedDepthFeature class
///
class SpatialInvariantColorFeature
{
public:
    enum DESCRIPTOR_TYPE
    {
        D_TYPE_BEEHIVE = 0,
        D_TYPE_ANNULAR,
        D_TYPE_HISTOGRAM,
        D_TYPE_CUBE3,
        D_TYPE_SURF  ,
        D_TYPE_BRIEF  ,
        D_TYPE_ORB    ,
        D_TYPE_BRISK ,
        D_TYPE_EMPTY= 255
    }patch_type_;
    cv::Mat rgb_img_;
    cv::Mat depth_16U_;                             //unit:mm
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_;  //unit:mm
    pcl::PointCloud<pcl::Normal>::Ptr normals_;     //unit:mm
    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree_;     //the Kd-tree for cloud_
    pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB,pcl::Normal>::Ptr normal_est_;
    double camera_fx, camera_fy,camera_cx,camera_cy;//彩色摄像机内参
    double _1_camera_fx, _1_camera_fy;            //减少计算量, 先计算出 1/camera_fx 和 1/camera_fy
    int _256_camera_fx, _256_camera_fy;            //减少计算量, 先计算出 256/camera_fx 和 256/camera_fy

    SpatialInvariantColorFeature(const uint _max_keypoints=200, const uint _patch_radius=40, const DESCRIPTOR_TYPE _method=D_TYPE_BEEHIVE);
    bool prepareFrame( const cv::Mat _rgb_image, const cv::Mat _depth_16U );
    uint extractFeaturePatch(const cv::Point& _pt, cv::Point3f &_pt3d, cv::Mat& _feature_patch, cv::Vec4d &_plane_coef, double &_plane_err, const uint &SPATIAL_RADIUS );

    uint calcPt6d(const cv::Point& _pt, cv::Point3f &_pt3d, cv::Vec4d &_plane_coef, double &_plane_err );
    uint calcPt6dSVD(const cv::Point& _pt, cv::Point3f &_pt3d, cv::Vec4d &_plane_coef, double &_plane_err, const uint &SPATIAL_RADIUS );
//    bool filterFakePoint( const cv::Point& _pt, const cv::Point3f &_pt3d, const cv::Vec4d _plane_coef );
    cv::Mat_<bool> extractImgPatch(const cv::Point3f &_pt3d, const uint &SPATIAL_RADIUS );
    uint warpPerspectivePatch( const cv::Point3f &_pt3d, const cv::Vec4d _plane_coef, cv::Mat &_feature_patch, const uint &SPATIAL_RADIUS );
    uint sampleCubeEvenly( const cv::Point3f &_pt3d, const cv::Vec4d _plane_coef, std::vector<cv::Vec3i> &_cube, const uint &SPATIAL_RADIUS, const double &_main_angle=0 );
    std::vector<cv::Vec3i> PyramidCube(const std::vector<cv::Vec3i> &_cube_hi_res );
    uint calcFeatureDir(const cv::Mat& _feature_patch, cv::Point2d &_main_dir, const double& _dense_thresh=0.2);
    uint generateFeatureCode(const cv::Mat& _feature_patch, const cv::Point2d &_main_dir, cv::Mat& _color_code, const double& _dense_thresh=0.2);
    uint generateFeatureCode_hov(const cv::Mat& _feature_patch, cv::Mat &_color_code, const uchar& _method=0);
    cv::Mat process(std::vector<cv::KeyPoint> &m_keypoints);
    cv::Mat processFPFH(std::vector<cv::KeyPoint> &m_keypoints, const uint &SPATIAL_RADIUS=70 );
    void match( const cv::Mat& queryDescriptors, const cv::Mat& trainDescriptors, std::vector<cv::DMatch>& matches ) const;
    bool restore_descriptor(const cv::Mat& _descriptor);
    uint MAX_KEYPOINTS;             //只保留最多这么些个特征点(考虑时间效率的原因)
    std::vector<cv::Point3f> keypoints_3D_;         //unit:mm
    cv::Mat descriptors_;                           //临时存放过滤后的鲁棒特征点特征描述子保存
//private:
    uint height;
    uint width;
    const uint PATCH_SIZE;          //提取图像片时的像素边长
    uint SPATIAL_R;                //单位mm,大于此宽度的断层，则认为是前、后景分离
    std::vector<cv::KeyPoint> keypoints_filtered_;  //临时存放过滤后的鲁棒特征点
public:
    ColorCoding color_encoder_;

    //////将图像片分成几块，每块分别进行颜色编码/////////////
    class PatchMask
    {
    public:
        uint ANGLE_RES;
        uint TOTAL_CELLS;
        const uchar BAD_CELL;
        PatchMask( const uint& _angle_res=12, const uint& _total_cells=64 ) : ANGLE_RES(_angle_res), TOTAL_CELLS(_total_cells), BAD_CELL(0xff) {}
        virtual const uchar getCellID(const uint& _xp, const uint& _yp, const uint& _main_angle_deg = 0 ) const = 0;
        virtual ~PatchMask(){}
    };

    class AnnularMask : public PatchMask
    {
    public:
        AnnularMask(const uint &_patch_size, const uint &_angle_res, const uint &_dist_res);
        const uint DIST_RES;        //对图像片进行分块编码时的距离分辨率
        std::vector<cv::Point2d> DIR_PER_CELL;//将patch分成小块后，每块对应的方向向量
        const uchar getCellID(const uint& _xp, const uint& _yp, const uint& _main_angle_deg = 0 ) const;
    private:
        std::vector<double> TAN_ANGLE_THRESH;//对应于每个角度分辨率的tan值, 用于判断某个点是在哪个角度量化值内
        cv::Mat mask_;//将特征图像片分成 ANGLE_RES*DIST_RES 块, 并编号为1~ANGLE_RES*DIST_RES, 中心和无效区域标号为0xff
    };
    class BeehiveMask : public PatchMask
    {
    public:
//        BeehiveMask(const uint &_patch_size);
        BeehiveMask(const uint &_patch_size, const uint &_layers=4, const uint &_angle_res=6*2);
        const uchar getCellID(const uint& _xp, const uint& _yp, const uint& _main_angle_deg = 0 ) const;
    private:
        std::vector<cv::Mat> masks_;//将特征图像片分成 61 块, 并编号为0~60, 无效区域标号为0xff, 旋转成一组
        cv::Mat rotateCellID_;
    };
    boost::shared_ptr<PatchMask> patch_mask_;
    boost::shared_ptr<AnnularMask> annular_mask_;
    boost::shared_ptr<BeehiveMask> beehive_mask_;
    ///////////////////////////////////////////////////
    class CubeMask
    {
    public:
        const int SIDE_LENGTH;
        enum CELL_FLAG
        {
            CELL_SHADOW = -1,
            CELL_SPACE = 0,
            CELL_SOLID = 1
        };

        CubeMask();
        const uchar getCellID(const uint& _xp, const uint& _yp, const uint& _zp, const uint& _main_angle_deg = 0 ) const;
    private:
        std::vector<uint32_t> cube3d_;

    };

    //调试用于显示
    bool DRAW_IMAG;             //是否构造并绘制显示图像
    cv::Mat rgb_show_;          //特征点被画在此图像上
    cv::Mat features_show_;     //特征图像片被排列显示
    cv::Mat features_restore_;  //特征描述子被重构成图像
};

#endif
