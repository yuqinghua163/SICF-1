#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/filesystem.hpp>

#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <std_msgs/String.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <image_transport/camera_subscriber.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/keypoints/narf_keypoint.h>
#include <pcl/features/narf.h>
#include <pcl/features/pfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/range_image_border_extractor.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/range_image/range_image.h>

#include "spatialinvariantcolorfeature.h"

namespace fs=boost::filesystem;
std::vector<boost::filesystem::path> getDirFiles( const std::string &_dir)
{
    std::vector<fs::path> files;
    std::vector<std::string> names;
    fs::recursive_directory_iterator end_iter;
    for(fs::recursive_directory_iterator iter(_dir);iter!=end_iter;iter++)
    {
        if ( !fs::is_directory( *iter ) )
            names.push_back( iter->path().string() ) ;
    }
    std::sort( names.begin(), names.end() );
    for( std::vector<std::string>::iterator it = names.begin(); it != names.end(); it++)
    {
        files.push_back( fs::path(*it) );
    }
    return files;
}
Eigen::Matrix4f
computeTfByPos7( std::vector<double> _pos_from, std::vector<double> _pos_to )
{
    Eigen::Matrix3f R_from_inv;
    double qw=_pos_from[0], qx=_pos_from[1], qy=_pos_from[2], qz=_pos_from[3];
    R_from_inv(0,0) = 1 - 2 * qy * qy - 2 * qz * qz;
    R_from_inv(0,1) =     2 * qx * qy - 2 * qz * qw;
    R_from_inv(0,2) =     2 * qx * qz + 2 * qy * qw;
    R_from_inv(1,0) =     2 * qx * qy + 2 * qz * qw;
    R_from_inv(1,1) = 1 - 2 * qx * qx - 2 * qz * qz;
    R_from_inv(1,2) =     2 * qy * qz - 2 * qx * qw;
    R_from_inv(2,0) =     2 * qx * qz - 2 * qy * qw;
    R_from_inv(2,1) =     2 * qy * qz + 2 * qx * qw;
    R_from_inv(2,2) = 1 - 2 * qx * qx - 2 * qy * qy;
    Eigen::Vector3f T_from_inv( _pos_from[4], _pos_from[5], _pos_from[6] );

    Eigen::Matrix3f R_to;
    qw=-_pos_to[0], qx=_pos_to[1], qy=_pos_to[2], qz=_pos_to[3];
    R_to(0,0) = 1 - 2 * qy * qy - 2 * qz * qz;
    R_to(0,1) =     2 * qx * qy - 2 * qz * qw;
    R_to(0,2) =     2 * qx * qz + 2 * qy * qw;
    R_to(1,0) =     2 * qx * qy + 2 * qz * qw;
    R_to(1,1) = 1 - 2 * qx * qx - 2 * qz * qz;
    R_to(1,2) =     2 * qy * qz - 2 * qx * qw;
    R_to(2,0) =     2 * qx * qz - 2 * qy * qw;
    R_to(2,1) =     2 * qy * qz + 2 * qx * qw;
    R_to(2,2) = 1 - 2 * qx * qx - 2 * qy * qy;
    Eigen::Vector3f T_to( _pos_to[4], _pos_to[5], _pos_to[6] );
    T_to = -R_to*T_to;

    Eigen::MatrixXf tf(R_to*R_from_inv);
    tf.conservativeResize(4,4);
                                           tf(0,3) = (R_to*T_from_inv)[0] + T_to[0];
                                           tf(1,3) = (R_to*T_from_inv)[1] + T_to[1];
                                           tf(2,3) = (R_to*T_from_inv)[2] + T_to[2];
    tf(3,0) = 0; tf(3,1) = 0; tf(3,2) = 0; tf(3,3) = 1;
    return tf;

}

template<class PointT> bool
regeisterCloud( pcl::PointCloud<PointT> cloud_src, pcl::PointCloud<PointT> cloud_tgt, Eigen::Matrix4f &tf )
{
    const uint min_res    = 15;//mm
    const uint norm_r     = 75;
    const uint feature_r  = 200;
    const uint SAC_Thresh = 15;

    /// 1. reprocess
    //remove NAN-Points
    std::vector<int> indices1,indices2;
    pcl::removeNaNFromPointCloud (cloud_src, cloud_src, indices1);
    pcl::removeNaNFromPointCloud (cloud_tgt, cloud_tgt, indices2);
    //Downsampling
    pcl::PointCloud<PointT> ds_src;
    pcl::PointCloud<PointT> ds_tgt;
    pcl::VoxelGrid<PointT> grid;
    grid.setLeafSize (min_res, min_res, min_res);//mm
    grid.setInputCloud ( cloud_src.makeShared());
    grid.filter (ds_src);
    grid.setInputCloud ( cloud_tgt.makeShared() );
    grid.filter (ds_tgt);
    // Normal-Estimation
    pcl::PointCloud<pcl::Normal>::Ptr norm_src (new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::Normal>::Ptr norm_tgt (new pcl::PointCloud<pcl::Normal>);
    boost::shared_ptr<pcl::search::KdTree<PointT> > tree_src ( new pcl::search::KdTree<PointT> );
    boost::shared_ptr<pcl::search::KdTree<PointT> > tree_tgt ( new pcl::search::KdTree<PointT> );
    pcl::NormalEstimation<PointT, pcl::Normal> ne;
    ne.setInputCloud ( ds_src.makeShared() );
    ne.setSearchSurface ( cloud_src.makeShared() );
    ne.setSearchMethod ( tree_src );
    ne.setRadiusSearch ( norm_r );//mm
    ne.compute (*norm_src);
    ne.setInputCloud (ds_tgt.makeShared());
    ne.setSearchSurface (cloud_tgt.makeShared());
    ne.setSearchMethod (tree_tgt);
    ne.setRadiusSearch ( norm_r );
    ne.compute (*norm_tgt);

    /// 2. Keypoints NARF
    pcl::RangeImage range_src;
    pcl::RangeImage range_tgt;
    //Range Image
    float angularResolution = (float) (  0.2f * (M_PI/180.0f));  //   0.5 degree in radians
    float maxAngleWidth     = (float) (360.0f * (M_PI/180.0f));  // 360.0 degree in radians
    float maxAngleHeight    = (float) (180.0f * (M_PI/180.0f));  // 180.0 degree in radians
    Eigen::Affine3f sensorPose = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);
    pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;
    float noiseLevel = 0.00;
    float minRange = 0.0f;
    int borderSize = 1;
    range_src.createFromPointCloud (cloud_src, angularResolution, maxAngleWidth, maxAngleHeight, sensorPose, coordinate_frame, noiseLevel, minRange, borderSize);
    range_tgt.createFromPointCloud (cloud_tgt, angularResolution, maxAngleWidth, maxAngleHeight, sensorPose, coordinate_frame, noiseLevel, minRange, borderSize);
    //Extract NARF-Keypoints
    pcl::RangeImageBorderExtractor range_image_ba;
    float support_size = min_res; //mm
    pcl::NarfKeypoint narf_keypoint_src (&range_image_ba);
    narf_keypoint_src.setRangeImage (&range_src);
    narf_keypoint_src.getParameters ().support_size = support_size;
    pcl::PointCloud<int> keypoints_ind_src;
    narf_keypoint_src.compute (keypoints_ind_src);
    pcl::NarfKeypoint narf_keypoint_tgt (&range_image_ba);
    narf_keypoint_tgt.setRangeImage (&range_tgt);
    narf_keypoint_tgt.getParameters ().support_size = support_size;
    pcl::PointCloud<int> keypoints_ind_tgt;
    narf_keypoint_tgt.compute (keypoints_ind_tgt);
    //get Keypoints as cloud
    pcl::PointCloud<PointT> keypoints_src;
    pcl::PointCloud<PointT> keypoints_tgt;
    keypoints_src.width = keypoints_ind_src.points.size();
    keypoints_src.height = 1;
    keypoints_src.is_dense = false;
    keypoints_src.points.resize (keypoints_src.width * keypoints_src.height);
    keypoints_tgt.width = keypoints_ind_tgt.points.size();
    keypoints_tgt.height = 1;
    keypoints_tgt.is_dense = false;
    keypoints_tgt.points.resize (keypoints_tgt.width * keypoints_tgt.height);
    for (size_t i = 0; i < keypoints_ind_src.points.size(); i++)
    {
        const int &ind_count = keypoints_ind_src.points[i];
        keypoints_src.points[i].x = range_src.points[ind_count].x;
        keypoints_src.points[i].y = range_src.points[ind_count].y;
        keypoints_src.points[i].z = range_src.points[ind_count].z;
    }
    for (size_t i = 0; i < keypoints_ind_tgt.points.size(); i++)
    {
        const int &ind_count = keypoints_ind_tgt.points[i];
        keypoints_tgt.points[i].x = range_tgt.points[ind_count].x;
        keypoints_tgt.points[i].y = range_tgt.points[ind_count].y;
        keypoints_tgt.points[i].z = range_tgt.points[ind_count].z;
    }
    std::cout << "NARF keypoints num=" << keypoints_src.size() << " " << keypoints_tgt.size() << std::endl;
    /// 3. Feature-Descriptor
    pcl::PFHEstimation<PointT, pcl::Normal, pcl::PFHSignature125> pfh_est_src;
    boost::shared_ptr<pcl::search::KdTree<PointT> > tree_pfh_src ( new pcl::search::KdTree<PointT> );
    pfh_est_src.setSearchMethod (tree_pfh_src);
    pfh_est_src.setRadiusSearch ( feature_r );//mm
    pfh_est_src.setSearchSurface (ds_src.makeShared());
    pfh_est_src.setInputNormals (norm_src);
    pfh_est_src.setInputCloud (keypoints_src.makeShared());
    pcl::PointCloud<pcl::PFHSignature125>::Ptr pfh_src (new pcl::PointCloud<pcl::PFHSignature125>);
    pfh_est_src.compute (*pfh_src);
    pcl::PFHEstimation<PointT, pcl::Normal, pcl::PFHSignature125> pfh_est_tgt;
    boost::shared_ptr<pcl::search::KdTree<PointT> > tree_pfh_tgt ( new pcl::search::KdTree<PointT> );
    pfh_est_tgt.setSearchMethod (tree_pfh_tgt);
    pfh_est_tgt.setRadiusSearch ( feature_r );//mm
    pfh_est_tgt.setSearchSurface (ds_tgt.makeShared());
    pfh_est_tgt.setInputNormals (norm_tgt);
    pfh_est_tgt.setInputCloud (keypoints_tgt.makeShared());
    pcl::PointCloud<pcl::PFHSignature125>::Ptr pfh_tgt (new pcl::PointCloud<pcl::PFHSignature125>);
    pfh_est_tgt.compute (*pfh_tgt);

    /// 4. Match
    Eigen::Matrix4f transformation;
    // Correspondence Estimation
    pcl::registration::CorrespondenceEstimation<pcl::PFHSignature125, pcl::PFHSignature125> corEst;
    corEst.setInputSource (pfh_src);
    corEst.setInputTarget (pfh_tgt);
    boost::shared_ptr<pcl::Correspondences> cor_all_ptr (new pcl::Correspondences);
    corEst.determineCorrespondences (*cor_all_ptr);
    //SAC
    pcl::registration::CorrespondenceRejectorSampleConsensus<PointT> sac;
    boost::shared_ptr<pcl::Correspondences> cor_inliers_ptr (new pcl::Correspondences);
    sac.setInputSource (keypoints_src.makeShared());
    sac.setInputTarget (keypoints_tgt.makeShared());
    sac.setInlierThreshold ( SAC_Thresh );
    sac.setMaximumIterations (100);
    sac.setInputCorrespondences (cor_all_ptr);
    sac.getCorrespondences (*cor_inliers_ptr);
    transformation = sac.getBestTransformation();

    /// end
//    pcl::PointCloud<PointT> cloud_tmp;
//    pcl::transformPointCloud (*cloud_src, *cloud_tmp, transformation);
    tf = transformation;

    return true;
}

std::vector<cv::DMatch>
getMatchsByTransform( const std::vector<cv::Point3f> &_from, const std::vector<cv::Point3f> &_to, const Eigen::Matrix4f &_trans, const bool _cross_check = false, const double dis_thresh=20  )
{

    std::vector<cv::DMatch> matches;     //new matches
    matches.reserve( _from.size() );
    for(int i=0; i<_from.size(); i++ )
    {
        Eigen::Vector4f pt_from( _from[i].x, _from[i].y, _from[i].z, 1 );
        pt_from = _trans*pt_from;
        int min_dist = INFINITY;
        int min_id_to = -1;
        for(int j=0; j<_to.size(); j++ )
        {
            Eigen::Vector4f pt_to( _to[j].x, _to[j].y, _to[j].z, 1 );
            double temp_dist = (pt_from-pt_to).norm();
            if( temp_dist < min_dist )
            {
                min_dist = temp_dist;
                min_id_to = j;
            }
         }
        if( min_id_to != -1 && min_dist < dis_thresh)//mm
            matches.push_back( cv::DMatch( i, min_id_to, min_dist) );
    }
    if( ! _cross_check )
        return matches;
    ///cross check
    Eigen::Matrix4f trans_inv = _trans.inverse();
    for( std::vector<cv::DMatch>::iterator p_match = matches.begin(); p_match != matches.end();  )
    {
        Eigen::Vector4f pt_to( _to[p_match->trainIdx].x, _to[p_match->trainIdx].y, _to[p_match->trainIdx].z, 1 );
        pt_to = trans_inv * pt_to;
        uint min_dist = p_match->distance;
        bool reject = false;
        for(int id_from=0; id_from<_from.size(); id_from++ )
        {
            Eigen::Vector4f pt_from( _from[id_from].x, _from[id_from].y, _from[id_from].z, 1 );

            if( id_from != p_match->queryIdx )
            if( min_dist >= (pt_from-pt_to).norm() )
            {
                reject = true;
                break;
            }
        }
        if( reject )
            p_match = matches.erase( p_match );
        else
            p_match ++;
    }
    return matches;
}

std::vector<cv::DMatch>
filterMatchsByTransform( const std::vector<cv::Point3f> &_from, const std::vector<cv::Point3f> &_to, const std::vector<cv::DMatch> _matches, const Eigen::Matrix4f &_trans, const double dis_thresh=50  )
{
    std::vector<cv::DMatch> matches;     //new matches
    matches.reserve( _matches.size() );
    for( std::vector<cv::DMatch>::const_iterator p_match = _matches.begin(); p_match != _matches.end(); p_match++ )
    {
        Eigen::Vector4f pt_from( _from[p_match->queryIdx].x, _from[p_match->queryIdx].y, _from[p_match->queryIdx].z, 1 );
        Eigen::Vector4f pt_to  ( _to  [p_match->trainIdx].x, _to  [p_match->trainIdx].y, _to  [p_match->trainIdx].z, 1 );
        if( (_trans*pt_from-pt_to).norm() <= dis_thresh )
            matches.push_back( *p_match );
    }
    return matches;
}

std::vector<cv::DMatch>
getMatchsByhomography( const std::vector<cv::KeyPoint> &_from, const std::vector<cv::KeyPoint> &_to, const cv::Mat& homography, const bool _cross_check = false )
{
    assert( homography.type() == CV_64F );
    std::vector<cv::DMatch> matches;     //new matches
    matches.reserve( _from.size() );
    std::vector<cv::Point2f> pts_from( _from.size() );
    for(int i=0; i<_from.size(); i++ )
        pts_from[i] = _from[i].pt;
    std::vector<cv::Point2f> pts_from_t;
    cv::perspectiveTransform( pts_from, pts_from_t, homography);

    for(int i=0; i<_from.size(); i++ )
    {
        cv::Point2f &pt_from = pts_from_t[i];
        std::cout << _from[i].pt << "->" << pt_from <<std::endl;
        int min_dist = INFINITY;
        int min_id_to = -1;
        for(int j=0; j<_to.size(); j++ )
        {
            const cv::Point2f & pt_to = _to[j].pt;
            double temp_dist = hypot( pt_from.x-pt_to.x, pt_from.y-pt_to.y );
            if( temp_dist < min_dist )
            {
                min_dist = temp_dist;
                min_id_to = j;
            }
         }
        if( min_id_to != -1 )//mm
            matches.push_back( cv::DMatch( i, min_id_to, min_dist) );
    }
    if( ! _cross_check )
        return matches;
    ///cross check
    cv::Mat homography_R = cv::Mat(homography, cv::Rect(0,0,2,2));
    cv::Mat homo_inv_R = homography_R.inv();
    double  homo_inv_T[2] = { -homography.at<double>(0,2), -homography.at<double>(1,2) };
    for( std::vector<cv::DMatch>::iterator p_match = matches.begin(); p_match != matches.end();  )
    {
        const cv::Point2f & pt = _to[p_match->trainIdx].pt;
        cv::Point2f pt_to;
        pt_to.x = pt.x * homo_inv_R.at<double>(0,0) + pt.y * homo_inv_R.at<double>(0,1) + homo_inv_T[0];
        pt_to.y = pt.x * homo_inv_R.at<double>(1,0) + pt.y * homo_inv_R.at<double>(1,1) + homo_inv_T[1];
        uint min_dist = p_match->distance;
        bool reject = false;
        for(int id_from=0; id_from<_from.size(); id_from++ )
        {
            const cv::Point2f & pt_from = _from[id_from].pt;
            if( id_from != p_match->queryIdx )
            if( min_dist >= hypot( pt_from.x-pt_to.x, pt_from.y-pt_to.y ) )
            {
                reject = true;
                break;
            }
        }
        if( reject )
            p_match = matches.erase( p_match );
        else
            p_match ++;
    }
    return matches;
}

std::vector<cv::DMatch> refineMatchesWithHomography
(
        const std::vector<cv::KeyPoint>& _src,
        const std::vector<cv::KeyPoint>& _dst,
        float reprojectionThreshold,
        const std::vector<cv::DMatch> matches,
        cv::Mat& homography
        )
{
    const int minNumberMatchesAllowed = 8;
    std::vector<cv::DMatch> inliers;
    if (matches.size() < minNumberMatchesAllowed)
        return inliers;
    // Prepare data for cv::findHomography
    std::vector<cv::Point2f> srcPoints(matches.size());
    std::vector<cv::Point2f> dstPoints(matches.size());
    for (size_t i = 0; i < matches.size(); i++)
    {
        srcPoints[i] = _src[matches[i].trainIdx].pt;
        dstPoints[i] = _dst[matches[i].queryIdx].pt;
    }
    // Find homography matrix and get inliers mask
    std::vector<unsigned char> inliersMask(srcPoints.size());
    homography = cv::findHomography(srcPoints,
                                    dstPoints,
                                    CV_FM_RANSAC,
                                    reprojectionThreshold,
                                    inliersMask);
    for (size_t i=0; i<inliersMask.size(); i++)
    {
        if (inliersMask[i])
            inliers.push_back(matches[i]);
    }
    return inliers;
}

int main(int argc, char **argv)
{
    fs::path filepath( "rgbd-data/scene4" );//desk     my_field scene4
    double depth_scale = 0.1;//cm->mm
    const bool COMPARE_TOGETHER = true;
    cv::initModule_nonfree();
    if( COMPARE_TOGETHER )
        std::cerr << "Warning! Use compare mode." << std::endl;
    pcl::console::setVerbosityLevel( pcl::console::L_ALWAYS );
    enum DescriptorType
    {
        DT_SICF,
        DT_BRIEF,
        DT_ORB,
        DT_BRISK,
        DT_FREAK,
        DT_SURF
    }descriptor_type = DT_SICF;
    SpatialInvariantColorFeature::DESCRIPTOR_TYPE SICF_TYPE = SpatialInvariantColorFeature::D_TYPE_BEEHIVE;

    for(int i=1;i<argc;++i)
    {
        if(strcmp(argv[i], "sicf") == 0 )
            descriptor_type = DT_SICF;
        if(strcmp(argv[i], "sicf-annular") == 0 )
            descriptor_type = DT_SICF,
            SICF_TYPE = SpatialInvariantColorFeature::D_TYPE_ANNULAR;
        if(strcmp(argv[i], "sicf-brisk") == 0 )
            descriptor_type = DT_SICF,
            SICF_TYPE = SpatialInvariantColorFeature::D_TYPE_BRISK;
        if(strcmp(argv[i], "sicf-surf") == 0 )
            descriptor_type = DT_SICF,
            SICF_TYPE = SpatialInvariantColorFeature::D_TYPE_SURF;
        if(strcmp(argv[i], "sicf-orb") == 0 )
            descriptor_type = DT_SICF,
            SICF_TYPE = SpatialInvariantColorFeature::D_TYPE_ORB;
        if(strcmp(argv[i], "sicf-hist") == 0 )
            descriptor_type = DT_SICF,
            SICF_TYPE = SpatialInvariantColorFeature::D_TYPE_HISTOGRAM;
        else if(strcmp(argv[i], "orb") == 0 || strcmp(argv[i], "ORB") ==0 )
            descriptor_type = DT_ORB;
        else if(strcmp(argv[i], "brief") == 0 || strcmp(argv[i], "BRIEF") ==0 )
            descriptor_type = DT_BRIEF;
        else if(strcmp(argv[i], "surf") == 0 || strcmp(argv[i], "SURF") ==0 )
            descriptor_type = DT_SURF;
        else if(strcmp(argv[i], "brisk") == 0 || strcmp(argv[i], "BRIEF") ==0 )
            descriptor_type = DT_BRISK;
        else if(strcmp(argv[i], "freak") == 0 || strcmp(argv[i], "FREAK") ==0 )
            descriptor_type = DT_FREAK;
    }

//    cv::StarDetector StarDetector;
//    cv::FastFeatureDetector FastFeatureDetector;
//    cv::GFTTDetector GFTTDetector( 300, 0.03, 2 );

    cv::SURF Surf;
    cv::BriefDescriptorExtractor BriefExtractor;
    cv::BRISK BRISK;
    cv::FREAK FreakExtractor;
    cv::ORB ORB;
    SpatialInvariantColorFeature sicf(1000,40,SICF_TYPE);

    cv::Mat rgb_last;
    pcl::PointCloud<pcl::PointXYZRGB> cloud_last;
    std::vector<cv::KeyPoint> keypoints_last;
    std::vector<cv::Point3f> keypoints3_last;
    cv::Mat descriptors_last;
    cv::Mat features_show_last;
    cv::Mat features_restore_last;
    bool RESTORE_PATCH = descriptor_type == DT_SICF
                    && ( sicf.patch_type_==SpatialInvariantColorFeature::D_TYPE_BEEHIVE
                      || sicf.patch_type_==SpatialInvariantColorFeature::D_TYPE_ANNULAR
                      || sicf.patch_type_==SpatialInvariantColorFeature::D_TYPE_CUBE3 );
    pcl::visualization::PCLVisualizer viewer_sicf("sicf key points");
    viewer_sicf.setCameraPosition(0,0,-400, 0,0,1, 0,-1,0);
    pcl::visualization::PCLVisualizer viewer_noob("noob key points");
    viewer_noob.setCameraPosition(0,0,-400, 0,0,1, 0,-1,0);
    pcl::visualization::PCLVisualizer viewer_norm("normals");
    viewer_norm.setCameraPosition(0,0,-400, 0,0,1, 0,-1,0);
    pcl::visualization::PCLVisualizer viewer_reg("reg_tf");
    viewer_reg.setCameraPosition(0,0,-400, 0,0,1, 0,-1,0);


    std::vector<fs::path> color_files = getDirFiles( filepath.string() + "/rgb" );
    std::vector<fs::path> depth_files = getDirFiles( filepath.string()+ "/depth" );
    assert( color_files.size()==depth_files.size() );
    std::vector<fs::path>::iterator path_color = color_files.begin();
    std::vector<fs::path>::iterator path_depth = depth_files.begin();
    std::ifstream groud_truth_pos( (filepath.string() + "/pos.txt").c_str() );
    std::vector<double> GTpos7_last(7);
    for( int data_cnt=1; path_color != color_files.end(); path_color++, path_depth++, data_cnt++)
    {
        ////////////////// 0. Load RGB-D data ///////////////
        cv::Mat rgb = cv::imread( path_color->string() );
        cv::Mat depth;
        assert( rgb.channels()==3 );
        if( strcmp(path_depth->extension().string().c_str(), ".yaml") == 0 )
        {
            cv::FileStorage fs( path_depth->string(), cv::FileStorage::READ);
            if( !fs.isOpened() ) break;
            fs["depth_16U"] >> depth;
            fs.release();
        }
        else
        {
            depth = cv::imread( path_depth->string(), cv::IMREAD_ANYDEPTH );
            depth *= depth_scale;
        }
        assert( depth.channels()==1 );
        std::vector<double> GTpos7(7);
        groud_truth_pos >> GTpos7[0] >> GTpos7[1] >> GTpos7[2] >> GTpos7[3] >> GTpos7[4] >> GTpos7[5] >> GTpos7[6];

        //////////// 1. Extrct key points & descriptor///////////////
        std::vector<cv::KeyPoint> keypoints0;
        cv::Mat descriptors;
        cv::Mat mask = cv::Mat::zeros(rgb.rows,rgb.cols,CV_8UC1);
        const int &BORDER = 50;
        cv::Mat(mask,cv::Rect(BORDER, BORDER, rgb.cols-2*BORDER, rgb.rows-2*BORDER)).setTo(1);
        BRISK.detect( rgb, keypoints0, mask );

        sicf.prepareFrame( rgb, depth );
        std::vector<cv::KeyPoint> keypoints;
        if( !COMPARE_TOGETHER && descriptor_type == DT_SICF )
            keypoints = keypoints0;
        else
        {
            ///filter keypoints by depth
            keypoints.reserve( keypoints0.size() );
            for( int i = 0;  i < keypoints0.size(); ++i )
            {
                const pcl::PointXYZRGB &pt = sicf.cloud_->at(keypoints0[i].pt.x,keypoints0[i].pt.y);
                if( pt.z!=0 && pt.z!=INFINITY )
                    keypoints.push_back( keypoints0[i] );
            }
            if(  COMPARE_TOGETHER )//filter key points, resrve valid for all
            {
                Surf.compute(  rgb, keypoints, descriptors );
                ORB.compute( rgb, keypoints, descriptors );
                BriefExtractor.compute( rgb, keypoints, descriptors);
                BRISK.compute( rgb, keypoints, descriptors );
                FreakExtractor.compute( rgb, keypoints, descriptors );
                sicf.process( keypoints );
            }
        }
        if( keypoints.size()<4 )
        {
            std::cerr << "Too few key points!!! = " << keypoints.size() << std::endl;
            continue;
        }

        switch (descriptor_type)
        {
        case DT_SURF:
            Surf.compute( rgb, keypoints, descriptors);
            break;
        case DT_ORB:
            ORB.compute( rgb, keypoints, descriptors);
            break;
        case DT_BRIEF:
            BriefExtractor.compute(rgb, keypoints, descriptors);
            break;
        case DT_BRISK:
            BRISK.compute(rgb, keypoints, descriptors);
            break;
        case DT_FREAK:
            FreakExtractor.compute(rgb, keypoints, descriptors);
            break;
        case DT_SICF:
            sicf.prepareFrame( rgb, depth );
            descriptors = sicf.process( keypoints );
            break;
        default:
            sicf.prepareFrame( rgb, depth );
            descriptors = sicf.processFPFH( keypoints );
            break;
        }
        if( keypoints.size()<3 )
        {
            std::cerr << "Too few key points!!! = " << keypoints.size() << std::endl;
            continue;
        }

        std::vector<cv::Point3f> keypoints3;
        if( descriptor_type == DT_SICF || COMPARE_TOGETHER )
        {
            sicf.restore_descriptor( descriptors );
            keypoints3 = sicf.keypoints_3D_;
        }
        else
        {
            keypoints3.reserve( keypoints.size() );
            for( int i = 0;  i < keypoints.size(); ++i )
            {
                const pcl::PointXYZRGB &pt = sicf.cloud_->at(keypoints[i].pt.x,keypoints[i].pt.y);
                keypoints3.push_back( cv::Point3f(pt.x,pt.y,pt.z) );
            }
        }

        /////////////////// 2. Keypoint match ///////////////////////////
        std::vector<cv::DMatch> matches;
        std::vector<cv::DMatch> matches_correct;//correct matches
        std::vector<cv::DMatch> matches_GT;     //ground truth matches
        Eigen::Matrix4f trans_GT;
        if( data_cnt>1 )
        {
            /// 2.1 descriptor match
            if( descriptor_type == DT_SICF)
            {
                if( sicf.patch_type_ == SpatialInvariantColorFeature::D_TYPE_SURF
                  ||sicf.patch_type_ == SpatialInvariantColorFeature::D_TYPE_HISTOGRAM
                  ||sicf.descriptors_.type() == CV_32F
                  ||sicf.descriptors_.type() == CV_64F)
                {
                    cv::BFMatcher matcher(cv::NORM_L2, true);
                    matcher.match(descriptors_last,descriptors,matches);
                }
                else
                    sicf.match( descriptors_last, descriptors, matches );
            }
            else
            {
                if( descriptors.type() == CV_32F || descriptors.type() == CV_64F )
                {
                    cv::BFMatcher matcher(cv::NORM_L2, true);
                    matcher.match(descriptors_last,descriptors,matches);
                }
                else
                {
                    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
                    matcher.match(descriptors_last,descriptors,matches);
                }
            }

            /// 2.2 calculate ground truth
            const int SENSOR_ERR = 25;
//            regeisterCloud( cloud_last, *ccdf_.cloud_, trans_GT );
            trans_GT = computeTfByPos7( GTpos7_last, GTpos7 );
            trans_GT(0,3) *= 1000;// m -> mm
            trans_GT(1,3) *= 1000;
            trans_GT(2,3) *= 1000;
            matches_GT = getMatchsByTransform( keypoints3_last, keypoints3, trans_GT, true, SENSOR_ERR);

            /// 2.3 calculate performance
//            cv::Mat homography;
//            matches_correct = refineMatchesWithHomography( keypoints_last, keypoints, 7, matches, homography );
            matches_correct = filterMatchsByTransform( keypoints3_last, keypoints3, matches, trans_GT, SENSOR_ERR );
            std::cout << data_cnt << ":\tCorrectRate=\t" << (double)matches_correct.size()/matches.size() <<  "\tRecallRate=\t" << (double)matches_correct.size()/matches_GT.size() << std::endl;
        }

        ////////////////// 3. Draw Key Points /////////////////////////
        cv::Mat keypoints_show = rgb.clone();
        if( descriptor_type != DT_SICF )
            cv::drawKeypoints( rgb, keypoints0, keypoints_show, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
        else
        {
            ///draw the same kepoints with the same color in different images/clouds
            pcl::PointCloud<pcl::PointXYZRGB> sicf_cloud(keypoints.size(),1);
            pcl::PointCloud<pcl::PointXYZRGB> noob_cloud(keypoints0.size(),1);
            cv::RNG rng;
            pcl::RGB color_draw;
            int cnt_sicf=0;
            int cnt_noob=0;
            for( int i = 0;  i < noob_cloud.size(); ++i )//draw original key points
            {
                color_draw.rgba = (uint32_t)rng.uniform(0x00000000,0x00ffffff);
                std::vector<cv::KeyPoint> cur_keypoint(1,keypoints0[i]);
                cv::drawKeypoints( rgb, cur_keypoint, keypoints_show, CV_RGB(color_draw.r,color_draw.g,color_draw.b), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS | cv::DrawMatchesFlags::DRAW_OVER_OUTIMG );
                const pcl::PointXYZRGB &noob_pt = sicf.cloud_->at(keypoints0[i].pt.x,keypoints0[i].pt.y);
                if( noob_pt.z!=0 && noob_pt.z!=INFINITY )
                {
                    noob_cloud.at(cnt_noob) = noob_pt;
                    noob_cloud.at(cnt_noob).rgba = color_draw.rgba;
                    cnt_noob++;
                }
                for( int j=0; j<keypoints.size(); j++)                    //draw corresponding SICF points
                    if( keypoints[j].pt == keypoints0[i].pt && cnt_sicf < sicf_cloud.size() )
                    {
                        sicf_cloud.at(cnt_sicf).x = keypoints3[j].x;
                        sicf_cloud.at(cnt_sicf).y = keypoints3[j].y;
                        sicf_cloud.at(cnt_sicf).z = keypoints3[j].z;
                        sicf_cloud.at(cnt_sicf).rgba = color_draw.rgba;
                        cnt_sicf++; break;
                    }
            }
            sicf_cloud.resize(cnt_sicf);
            noob_cloud.resize(cnt_noob);
            ///show all clouds
            if( data_cnt==1 )
            {
                viewer_sicf.addPointCloud( sicf.cloud_, "cloud0" );
                viewer_sicf.addPointCloud( sicf_cloud.makeShared(),"key_points" );
                viewer_sicf.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud0");
                viewer_sicf.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8, "key_points");
                viewer_noob.addPointCloud( sicf.cloud_, "cloud0" );
                viewer_noob.addPointCloud( noob_cloud.makeShared(),"key_points" );
                viewer_noob.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud0");
                viewer_noob.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8, "key_points");
            }
            else
            {
                viewer_sicf.updatePointCloud( sicf.cloud_, "cloud0" );
                viewer_sicf.updatePointCloud( sicf_cloud.makeShared(),"key_points" );
                viewer_noob.updatePointCloud( sicf.cloud_, "cloud0" );
                viewer_noob.updatePointCloud( noob_cloud.makeShared(),"key_points" );
            }
            viewer_norm.removeAllPointClouds();
            viewer_norm.addPointCloud( sicf.cloud_, "cloud0" );
            viewer_norm.addPointCloudNormals<pcl::PointXYZRGB,pcl::Normal>( sicf.cloud_, sicf.normals_, 20, 10, "normals" );
            cv::imshow("keypoints_sicf",sicf.rgb_show_);
        }
        cv::imshow("keypoints_noob",keypoints_show);
        ///show cloud_reg
        if( data_cnt > 1 )
        {
            pcl::PointCloud<pcl::PointXYZRGB> cloud_reg;
            pcl::transformPointCloud (cloud_last, cloud_reg, trans_GT);
            if( data_cnt ==2 )
            {
                viewer_reg.addPointCloud( sicf.cloud_,"points" );
                viewer_reg.addPointCloud( cloud_reg.makeShared(),"reg_points" );
            }
            else
            {
                viewer_reg.updatePointCloud( sicf.cloud_,"points" );
                viewer_reg.updatePointCloud( cloud_reg.makeShared(),"reg_points" );
            }
        }

        //////////////// 4. Draw maches //////////////
        cv::Mat img_matches;
        cv::Mat img_matches_correct;
        cv::Mat img_matches_patch;
        cv::Mat img_matches_patch_restore;
        cv::Mat img_matches_GT;
        if( data_cnt>1 )
        {
            cv::drawMatches( rgb_last, keypoints_last, rgb, keypoints, matches, img_matches );
            cv::drawMatches( rgb_last, keypoints_last, rgb, keypoints, matches_correct, img_matches_correct );
            cv::drawMatches( rgb_last, keypoints_last, rgb, keypoints, matches_GT, img_matches_GT );
            std::vector<cv::KeyPoint> keypoints_last_temp = keypoints_last;
            std::vector<cv::KeyPoint> keypoints_temp = keypoints;
            for(int i=0; i<keypoints_last_temp.size(); i++)
            {
                keypoints_last_temp[i].pt.x = (i%10)*(features_show_last.cols/10)+features_show_last.cols/10/2+1;
                keypoints_last_temp[i].pt.y = (i/10)*(features_show_last.rows/10)+features_show_last.rows/10/2+1;
            }
            for(int i=0; i<keypoints_temp.size(); i++)
            {
                keypoints_temp[i].pt.x = (i%10)*(sicf.features_show_.cols/10)+sicf.features_show_.cols/10/2+1;
                keypoints_temp[i].pt.y = (i/10)*(sicf.features_show_.rows/10)+sicf.features_show_.rows/10/2+1;
            }
            if( descriptor_type == DT_SICF )
            {
                cv::Mat temp_img, temp_img_last;
                cv::cvtColor( sicf.features_show_, temp_img, CV_RGBA2RGB);
                cv::cvtColor( features_show_last, temp_img_last, CV_RGBA2RGB);
                cv::drawMatches( temp_img_last, keypoints_last_temp, temp_img, keypoints_temp, matches, img_matches_patch );
                cv::line( img_matches_patch, cv::Point(img_matches_patch.cols/2,0), cv::Point(img_matches_patch.cols/2,img_matches_patch.rows), CV_RGB(255,255,255), 3);
                for( int i=0; i<matches.size(); i++ )
                {
                    std::stringstream dist_str;
                    cv::Point pt_temp;
                    dist_str << matches[i].distance;
                    pt_temp = keypoints_last_temp[ matches[i].queryIdx ].pt;
                    cv::putText( img_matches_patch, dist_str.str(), pt_temp, cv::FONT_HERSHEY_SIMPLEX, 0.7, CV_RGB(255,0,0) );
                    pt_temp = keypoints_temp[ matches[i].trainIdx ].pt;
                    pt_temp.x += img_matches_patch.cols/2;
                    cv::putText( img_matches_patch, dist_str.str(), pt_temp, cv::FONT_HERSHEY_SIMPLEX, 0.7, CV_RGB(255,0,0) );
                }
                if( RESTORE_PATCH )
                {
                    cv::cvtColor( sicf.features_restore_, temp_img, CV_RGBA2RGB);
                    cv::cvtColor( features_restore_last, temp_img_last, CV_RGBA2RGB);
                    cv::drawMatches( temp_img_last, keypoints_last_temp, temp_img, keypoints_temp, matches, img_matches_patch_restore );
                    cv::line( img_matches_patch_restore, cv::Point(img_matches_patch_restore.cols/2,0), cv::Point(img_matches_patch_restore.cols/2,img_matches_patch_restore.rows), CV_RGB(255,255,255), 3);
                    for( int i=0; i<matches.size(); i++ )
                    {
                        std::stringstream dist_str;
                        cv::Point pt_temp;
                        dist_str << matches[i].distance;
                        pt_temp = keypoints_last_temp[ matches[i].queryIdx ].pt;
                        cv::putText( img_matches_patch_restore, dist_str.str(), pt_temp, cv::FONT_HERSHEY_SIMPLEX, 0.7, CV_RGB(255,0,0) );
                        pt_temp = keypoints_temp[ matches[i].trainIdx ].pt;
                        pt_temp.x += img_matches_patch_restore.cols/2;
                        cv::putText( img_matches_patch_restore, dist_str.str(), pt_temp, cv::FONT_HERSHEY_SIMPLEX, 0.7, CV_RGB(255,0,0) );
                    }
                    cv::imshow("img_matches_restore",img_matches_patch_restore);
                }
                cv::imshow("img_matches_patch",img_matches_patch);
            }
            cv::imshow("img_matches",img_matches);
            cv::imshow("img_matches_correct",img_matches_correct);
            cv::imshow("img_matches_GT",img_matches_GT);
        }

        ////////////////////// 5. Save Image ///////////////////////////
        char key = cv::waitKey(10);
        while( -1==key )
        {
            key = cv::waitKey(5);
            viewer_sicf.spinOnce(5);
            viewer_noob.spinOnce(5);
            viewer_norm.spinOnce(5);
            viewer_reg.spinOnce(5);
        }
        if( key == 's' )
        {
            std::stringstream cnt_str;
            cnt_str << data_cnt;
            if( descriptor_type == DT_SICF )
            {
                cv::imwrite(filepath.string()+"/results/"+"keypoints_noob"+cnt_str.str()+".jpg",keypoints_show);
                cv::imwrite(filepath.string()+"/results/"+"keypoints_sicf"+cnt_str.str()+".jpg",sicf.rgb_show_);
                cv::imwrite(filepath.string()+"/results/"+"features_show"+cnt_str.str()+".jpg",sicf.features_show_);
            }
            if( data_cnt>1 )
            {
                cv::imwrite(filepath.string()+"/results/"+"img_matches"  +cnt_str.str()+".jpg",img_matches);
                cv::imwrite(filepath.string()+"/results/"+"img_matches_correct"+cnt_str.str()+".jpg",img_matches_correct);
            }
            if( RESTORE_PATCH )
                cv::imwrite(filepath.string()+"/results/"+"features_restore"+cnt_str.str()+".jpg",sicf.features_restore_);
            if( descriptor_type == DT_SICF )
                pcl::io::savePCDFile(filepath.string()+"/results/"+"pointcloud"+cnt_str.str()+".pcd",*sicf.cloud_);
        }
        else if( 27 == key )
            break;

        rgb_last = rgb.clone();
        pcl::copyPointCloud( *sicf.cloud_, cloud_last );
        keypoints_last = keypoints;
        keypoints3_last = keypoints3;
        descriptors_last = descriptors.clone();
        features_show_last = sicf.features_show_.clone();
        features_restore_last = sicf.features_restore_.clone();
        GTpos7_last = GTpos7;
    }

    groud_truth_pos.close();
    //     ros::spin();
    return 0;
}
