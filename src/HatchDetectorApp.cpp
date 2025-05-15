#include <HatchDetectorApp.hpp>

namespace fs = std::filesystem;
namespace
{
    // IP address of the robot's TCP server
    const char *SERVER_IP = "192.168.1.100";
    const int SERVER_PORT = 50000;
    constexpr int MESSAGE_SIZE = 64;
}

std::mutex pose_mutex;
cv::Point3d latest_position;
cv::Vec3d latest_orientation;

std::string formatFloat(float value, int width, int precision)
{
    std::ostringstream oss;
    char sign = value >= 0 ? '+' : '-';
    value = std::abs(value);

    int int_part = static_cast<int>(value);
    float frac_part = value - int_part;

    oss << sign
        << std::setfill('0') << std::setw(width - (1 + 1 + precision)) // width minus sign, dot, decimals
        << int_part
        << '.'
        << std::setw(precision) << static_cast<int>(frac_part * std::pow(10, precision));

    return oss.str();
}

int computeChecksum(const std::string &body)
{
    int sum = 0;
    for (char c : body.substr(0, 60))
    {
        sum += static_cast<unsigned char>(c);
    }
    return sum % 256;
}

// Starts a background TCP client that connects to a FANUC server.
// When the server sends 0x0001, the latest 6D pose is sent back as comma-separated ASCII.
void startTCPClient() {
    while (true) {
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) {
            std::cerr << "Socket creation failed" << std::endl;
            sleep(2);
            continue;
        }

        sockaddr_in server_addr{};
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(SERVER_PORT);
        inet_pton(AF_INET, SERVER_IP, &server_addr.sin_addr);

        std::cout << "Attempting to connect to FANUC..." << std::endl;
        if (connect(sock, (sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
            std::cerr << "Connection failed. Retrying..." << std::endl;
            close(sock);
            sleep(2);
            continue;
        }

        std::cout << "Connected to FANUC server." << std::endl;

        char recv_buffer[10] = {};
        int n = recv(sock, recv_buffer, 10, 0);
        if (n <= 0) {
            std::cerr << "Did not receive ping." << std::endl;
            close(sock);
            continue;
        }

        std::string ping(recv_buffer, n);
        if (ping.find("SEND DATA") == std::string::npos) {
            std::cerr << "Unexpected ping message: " << ping << std::endl;
            close(sock);
            continue;
        }

        pose_mutex.lock();
        std::string message;
        message += formatFloat(latest_position.x, 9, 4);
        message += formatFloat(latest_position.y, 9, 4);
        message += formatFloat(latest_position.z, 9, 4);
        message += formatFloat(latest_orientation[0], 9, 4);
        message += formatFloat(latest_orientation[1], 9, 4);
        message += formatFloat(latest_orientation[2], 9, 4);
        pose_mutex.unlock();

        while (message.size() < 60) message += ' ';

        int checksum = computeChecksum(message);
        if (checksum < 10) message += "00";
        else if (checksum < 100) message += "0";
        message += std::to_string(checksum);
        message += "#";

        while (message.size() < 64) message += ' ';

        std::cout << "Sending message (\"" << message << "\")" << std::endl;
        send(sock, message.c_str(), MESSAGE_SIZE, 0);

        memset(recv_buffer, 0, sizeof(recv_buffer));
        n = recv(sock, recv_buffer, 10, 0);
        if (n > 0) std::cout << "Ack: " << std::string(recv_buffer, n) << std::endl;

        close(sock);
        sleep(1);
    }
}

// Constructor: Set up camera initialization parameters for ZED
HatchDetectorApp::HatchDetectorApp()
{
    // Set resolution to 720p
    init_params.camera_resolution = sl::RESOLUTION::HD1200;

    // Use neural depth mode for better accuracy
    init_params.depth_mode = sl::DEPTH_MODE::NEURAL_PLUS;

    // Output coordinates in meters
    init_params.coordinate_units = sl::UNIT::METER;

    // Set FPS to 15 for smoother depth updates
    init_params.camera_fps = 15;

    // Start TCP communication in a background thread
    std::thread tcp_thread(startTCPClient);
    tcp_thread.detach();
}

// Destructor: Ensure camera is properly closed
HatchDetectorApp::~HatchDetectorApp()
{
    zed.close();
}

// Initialize & Attempt to open the ZED camera and return success status
bool HatchDetectorApp::initialize()
{
    auto err = zed.open(init_params);
    if (err != sl::ERROR_CODE::SUCCESS)
    {
        std::cerr << "ZED Camera open failed: " << sl::toString(err) << std::endl;
        return false;
    }

    detector = cv::ORB::create();
    matcher = cv::BFMatcher(cv::NORM_HAMMING);
    loadTemplates();

    return true;
}

void HatchDetectorApp::loadTemplates()
{
    std::string base_path = PROJECT_SOURCE_DIR;  // Defined by CMake

    fs::path template_path_1 = fs::path(base_path) / "data" / "templates" / "hatch_view1.png";
    fs::path template_path_2 = fs::path(base_path) / "data" / "templates" / "hatch_view2.png";

    std::vector<fs::path> paths = {template_path_1, template_path_2};

    for (const auto &path : paths)
    {
        cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);

        if (img.empty())
        {
            std::cerr << "Failed to load template: " << path << std::endl;
            continue;
        }

        TemplateData tpl;
        tpl.image = img;
        detector->detectAndCompute(img, cv::noArray(), tpl.keypoints, tpl.descriptors);
        templates.push_back(tpl);
    }
}

// Main loop: To continuously grab frames and apply hatch detection
void HatchDetectorApp::run()
{
    sl::Mat zed_image;
    sl::Mat point_cloud;

    while (true)
    {
        // Try to grab a new frame
        if (zed.grab(runtime_params) == sl::ERROR_CODE::SUCCESS)
        {
            // Retrieve the left image from ZED
            zed.retrieveImage(zed_image, sl::VIEW::LEFT);

            // Retrieve the full 3D point cloud from the ZED camera
            zed.retrieveMeasure(point_cloud, sl::MEASURE::XYZ);

            // Wrap it in a cv::Mat (OpenCV format)
            cv::Mat cv_image(zed_image.getHeight(), zed_image.getWidth(), CV_8UC4, zed_image.getPtr<sl::uchar1>(sl::MEM::CPU));

            // For drawing overlays
            cv::Mat display_image = cv_image.clone();

            cv::Mat gray;
            cv::cvtColor(cv_image, gray, cv::COLOR_RGBA2GRAY);
            matchTemplates(gray, display_image, point_cloud);

            // // Perform hatch detection and overlay results
            // detectHatches(display_image, cv_image, point_cloud);

            // Show the result
            cv::imshow("ZED View", display_image);

            // Exit loop on 'q'
            if (cv::waitKey(30) == 'q')
                break;
        }
    }
}

void HatchDetectorApp::matchTemplates(const cv::Mat &gray_image, cv::Mat &display_image, const sl::Mat &point_cloud)
{
    std::vector<cv::KeyPoint> frame_kp;
    cv::Mat frame_desc;
    detector->detectAndCompute(gray_image, cv::noArray(), frame_kp, frame_desc);
    std::cout << "Live frame keypoints: " << frame_kp.size() << std::endl;
    if (frame_desc.empty())
    {
        return;
    }

    // Show keypoints in the live frame
    cv::Mat keypoint_debug;
    cv::drawKeypoints(gray_image, frame_kp, keypoint_debug);
    cv::imshow("Live Keypoints", keypoint_debug);

    int best_match_count = 0;
    std::string matched_template_name;

    for (size_t i = 0; i < templates.size(); ++i)
    {
        const auto &tpl = templates[i];
        std::vector<cv::DMatch> matches;
        matcher.match(tpl.descriptors, frame_desc, matches);

        std::vector<cv::DMatch> good;
        for (auto &m : matches)
        {
            if (m.distance < 50.0)
            {
                good.push_back(m);
            }
        }

        if (good.size() < 5)
        {
            continue;
        }

        std::vector<cv::Point2f> ref_pts, frm_pts;
        for (auto &m : good)
        {
            ref_pts.push_back(tpl.keypoints[m.queryIdx].pt);
            frm_pts.push_back(frame_kp[m.trainIdx].pt);
        }

        cv::Mat mask;
        cv::Mat H = cv::findHomography(ref_pts, frm_pts, cv::RANSAC, 3.0, mask);
        if (H.empty())
        {
            continue;
        }

        std::vector<cv::Point2f> corners = {{0, 0}, {(float)tpl.image.cols, 0}, {(float)tpl.image.cols, (float)tpl.image.rows}, {0, (float)tpl.image.rows}};
        std::vector<cv::Point2f> projected;
        cv::perspectiveTransform(corners, projected, H);
        for (int j = 0; j < 4; ++j)
        {
            cv::line(display_image, projected[j], projected[(j + 1) % 4], cv::Scalar(255, 0, 0), 2);
        }

        cv::Rect bbox = cv::boundingRect(projected);
        auto valid3D = extract3DPoints(point_cloud, bbox);
        cv::Point3d centroid;
        cv::Vec3d euler;
        cv::Mat eigen;
        if (computePCAOrientation(valid3D, centroid, euler, &eigen))
        {
            applyTemporalSmoothing(centroid, euler);
            {
                std::lock_guard<std::mutex> lock(pose_mutex);
                latest_position = centroid;
                latest_orientation = euler;
            }
            draw3DAxes(display_image, centroid, eigen);
            drawPoseText(display_image, bbox, centroid, euler);
        }
        best_match_count = good.size();
        matched_template_name = "Template " + std::to_string(i + 1);

        // Debug: show template keypoints
        cv::Mat tpl_debug;
        cv::drawKeypoints(tpl.image, tpl.keypoints, tpl_debug);
        cv::imshow("Template " + std::to_string(i + 1), tpl_debug);

        break;
    }

    if (!matched_template_name.empty())
    {
        std::string overlay = matched_template_name + " | Matches: " + std::to_string(best_match_count);
        cv::putText(display_image, overlay, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    }
}

void HatchDetectorApp::draw3DAxes(cv::Mat &image, const cv::Point3d &centroid, const cv::Mat &eigen)
{
    auto calib = zed.getCameraInformation().camera_configuration.calibration_parameters.left_cam;
    cv::Point center2D(
        static_cast<int>((centroid.x * calib.fx) / centroid.z + calib.cx),
        static_cast<int>((centroid.y * calib.fy) / centroid.z + calib.cy));
    double axis_len = 0.2; // meters
    for (int i = 0; i < 3; ++i)
    {
        cv::Point3d end = centroid + axis_len * cv::Point3d(eigen.at<double>(i, 0), eigen.at<double>(i, 1), eigen.at<double>(i, 2));
        cv::Point end2D(
            static_cast<int>((end.x * calib.fx) / end.z + calib.cx),
            static_cast<int>((end.y * calib.fy) / end.z + calib.cy));
        cv::Scalar color = (i == 0) ? cv::Scalar(0, 0, 255) : (i == 1) ? cv::Scalar(0, 255, 0)
                                                                       : cv::Scalar(255, 0, 0);
        cv::line(image, center2D, end2D, color, 2);
    }
}

void HatchDetectorApp::drawPoseText(cv::Mat &image, const cv::Rect &bbox, const cv::Point3d &centroid, const cv::Vec3d &euler)
{
    char text[128];
    snprintf(text, sizeof(text),
             "T[%.2f,%.2f,%.2f] R[%.1f,%.1f,%.1f]",
             centroid.x, centroid.y, centroid.z,
             euler[0] * 180 / CV_PI, euler[1] * 180 / CV_PI, euler[2] * 180 / CV_PI);
    cv::putText(image, text, bbox.tl() + cv::Point(0, -10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 2);
}

// Preprocess the image to grayscale, blur, and detect edges for contour analysis
// Preprocess the input image and return the edge-detected image for contour extraction
// Converts RGBA to grayscale, equalizes histogram, blurs, and applies Canny + morphological close
void HatchDetectorApp::preprocessImage(cv::Mat &image, cv::Mat &output)
{
    cv::Mat gray, edges, blurred, morph;

    // Convert RGBA to grayscale for edge detection
    cv::cvtColor(image, gray, cv::COLOR_RGBA2GRAY);

    // Improve contrast with histogram equalization
    cv::equalizeHist(gray, gray);

    // Smooth image to reduce noise with Gaussian blur
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

    // Detect edges with Canny algorithm
    cv::Canny(blurred, edges, 10, 50);

    // Close small gaps in contours/edges
    cv::morphologyEx(edges, morph, cv::MORPH_CLOSE, cv::Mat(), cv::Point(-1, -1), 1);

    // Show intermediate stages for debuggingcv::imshow("Gray", gray);
    cv::imshow("Gray", gray);
    cv::imshow("Edges", edges);
    cv::imshow("Blurred", blurred);
    cv::imshow("Morph", morph);

    output = morph; // overwrite input image for simplicity
}

// Extract contours from the preprocessed image
std::vector<std::vector<cv::Point>> HatchDetectorApp::findContoursFromImage(const cv::Mat &edgeImage)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edgeImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    return contours;
}

// Validate whether a contour is likely a hatch (rectangle-like)
// Check whether a contour approximates a valid rectangular hatch shape.
// Applies convexity, area, solidity, and aspect ratio heuristics.
bool HatchDetectorApp::isValidContour(const std::vector<cv::Point> &approx, const std::vector<cv::Point> &contour)
{
    // Check for rectangular, convex shape with 4 vertices
    if (approx.size() != 4 || !cv::isContourConvex(approx))
        return false;

    double area = cv::contourArea(approx);
    if (area < 50)
    {
        // std::cout << "Rejected: area too small\n";
        return false;
    }

    // Check solidity to ensure shape is filled well
    std::vector<cv::Point> hull;
    cv::convexHull(contour, hull);
    double solidity = (cv::contourArea(hull) > 0) ? cv::contourArea(contour) / cv::contourArea(hull) : 0.0;
    if (solidity < 0.85)
    {
        // std::cout << "Rejected: low solidity\n";
        return false;
    }

    // Filter by reasonable aspect ratio (w/h)
    cv::RotatedRect rect = cv::minAreaRect(contour);
    float aspectRatio = std::max(rect.size.width, rect.size.height) / std::min(rect.size.width, rect.size.height);
    if (aspectRatio < 0.8 || aspectRatio > 3.0)
    {
        // std::cout << "Rejected: bad aspect ratio\n";
        return false;
    }
    return (aspectRatio >= 0.8 && aspectRatio <= 3.0);
}

// Extract valid 3D points from the point cloud inside a bounding box
std::vector<cv::Point3f> HatchDetectorApp::extract3DPoints(const sl::Mat &point_cloud, const cv::Rect &bbox)
{
    // Stores valid 3D points inside bbox
    std::vector<cv::Point3f> points;

    // Iterate through the bounding box region to collect valid 3D points
    for (int y = bbox.y; y < bbox.y + bbox.height; ++y)
    {
        for (int x = bbox.x; x < bbox.x + bbox.width; ++x)
        {
            // Skip out-of-bounds indice
            if (x < 0 || y < 0 || x >= point_cloud.getWidth() || y >= point_cloud.getHeight())
                continue;

            // Get 3D coordinates at pixel (x, y)
            sl::float4 pt;
            point_cloud.getValue(x, y, &pt);

            // Skip invalid or NaN measurements
            if (!std::isnan(pt.x) && !std::isnan(pt.y) && !std::isnan(pt.z) && pt.z > 0.0f)
            {
                points.emplace_back(pt.x, pt.y, pt.z);
            }
        }
    }
    return points;
}

// Helper to compute the 3D center point of a region using ZED depth
cv::Point3f HatchDetectorApp::compute3DCenter(const std::vector<cv::Point3f> &validPoints)
{
    // Return invalid result if no valid 3D points found
    if (validPoints.empty())
        return cv::Point3f(NAN, NAN, NAN);

    // Stores Euclidean distances for sorting
    std::vector<float> distances;
    distances.reserve(validPoints.size());

    // Compute Euclidean distance from origin to the 3D point
    for (const auto &p : validPoints)
    {
        float depth = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        distances.push_back(depth);
    }

    // Apply median filtering on the collected distances
    // Step 1: Create index map [0, 1, 2, ..., N-1]
    std::vector<size_t> indices(validPoints.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Step 2: Rearrange indices so that the middle index corresponds to the median depth
    std::nth_element(
        indices.begin(),
        indices.begin() + indices.size() / 2,
        indices.end(),
        [&](size_t i1, size_t i2)
        {
            return distances[i1] < distances[i2];
        });

    // Step 3: Use median index to retrieve the corresponding 3D point
    size_t medianIndex = indices[indices.size() / 2];
    return validPoints[medianIndex];
}

// Converts a 3x3 rotation matrix to Euler angles (roll, pitch, yaw) in radians
// Assumes intrinsic XYZ rotation order (roll around X, pitch around Y, yaw around Z)
cv::Vec3d HatchDetectorApp::rotationMatrixToEulerAngles(const cv::Mat &R)
{
    double sy = std::sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) +
                          R.at<double>(1, 0) * R.at<double>(1, 0));
    bool singular = sy < 1e-6;
    double x, y, z;
    if (!singular)
    {
        x = std::atan2(R.at<double>(2, 1), R.at<double>(2, 2)); // roll
        y = std::atan2(-R.at<double>(2, 0), sy);                // pitch
        z = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0)); // yaw
    }
    else
    {
        x = std::atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
        y = std::atan2(-R.at<double>(2, 0), sy);
        z = 0;
    }
    return cv::Vec3d(x, y, z);
}

// Helper to compute PCA centroid and orientation from a vector of 3D points
// Perform PCA on a 3D point set and extract centroid and Euler angles (XYZ intrinsic rotation order).
// Input: 3D points
// Output: centroid of the point set and orientation as roll/pitch/yaw (in radians)
bool HatchDetectorApp::computePCAOrientation(const std::vector<cv::Point3f> &points, cv::Point3d &centroid, cv::Vec3d &eulerAngles, cv::Mat *eigen_out)
{
    // Require at least 3 points
    if (points.size() < 3)
        return false;

    // Construct a matrix of 3D points with double precision for accuracy
    cv::Mat data(static_cast<int>(points.size()), 3, CV_64F);
    for (size_t i = 0; i < points.size(); ++i)
    {
        data.at<double>(i, 0) = points[i].x;
        data.at<double>(i, 1) = points[i].y;
        data.at<double>(i, 2) = points[i].z;
    }

    // Apply PCA to extract the major axes
    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);

    // Centroid of the point cloud
    centroid = cv::Point3d(pca.mean.at<double>(0),
                           pca.mean.at<double>(1),
                           pca.mean.at<double>(2));

    // Eigenvectors represent the orientation of the object
    cv::Mat eigen = pca.eigenvectors;

    // Provide eigenvectors to the caller if requested
    if (eigen_out)
        *eigen_out = eigen.clone();

    // Convert eigenvectors to rotation matrix for Euler extraction
    cv::Mat R = eigen.t();
    eulerAngles = rotationMatrixToEulerAngles(R);

    return true;
}

// Apply temporal smoothing to reduce jitter in position and orientation readings
void HatchDetectorApp::applyTemporalSmoothing(cv::Point3d &position, cv::Vec3d &orientation)
{
    const size_t max_history = 5;
    // Update position history
    position_history.push_back(position);
    if (position_history.size() > max_history)
        position_history.pop_front();

    // Update orientation history
    orientation_history.push_back(orientation);
    if (orientation_history.size() > max_history)
        orientation_history.pop_front();

    // Compute smoothed average for position
    cv::Point3d sum_pos(0, 0, 0);
    for (const auto &p : position_history)
        sum_pos += p;
    position = sum_pos * (1.0 / position_history.size());

    // Compute smoothed average for orientation (Euler angles)
    cv::Vec3d sum_rot(0, 0, 0);
    for (const auto &r : orientation_history)
        sum_rot += r;
    orientation = sum_rot * (1.0 / orientation_history.size());
}

// Detect hatch-like rectangles using edge detection and contour filtering
void HatchDetectorApp::detectHatches(cv::Mat &display_image, cv::Mat &original_image, const sl::Mat &point_cloud)
{
    // Prerocess Image (gray --> blurred --> edges --> morph)
    cv::Mat edge_img;
    preprocessImage(original_image, edge_img);

    // Extract contours from the images
    std::vector<std::vector<cv::Point>> contours = findContoursFromImage(edge_img);

    for (const auto &contour : contours)
    {
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx, 0.02 * cv::arcLength(contour, true), true);

        // Check for rectangular, solidity, aspect ratio (w/h)
        if (!isValidContour(approx, contour))
            continue;

        // Draw contour (green)
        cv::drawContours(display_image, std::vector<std::vector<cv::Point>>{approx}, -1, cv::Scalar(0, 255, 0), 2);

        // Draw bounding box (blue)
        cv::Rect bbox = cv::boundingRect(approx);
        cv::rectangle(display_image, bbox, cv::Scalar(255, 0, 0), 2);

        // Extract Valid 3D points for each contour
        std::vector<cv::Point3f> validPoints3D = extract3DPoints(point_cloud, bbox);

        // // Compute and project 3D center onto image
        // cv::Point3f center3D = compute3DCenter(validPoints3D);
        // if (!std::isnan(center3D.z))
        // {
        //     std::cout << "3D Center: [" << center3D.x << ", " << center3D.y << ", " << center3D.z << "]\n";
        //     auto calib = zed.getCameraInformation().camera_configuration.calibration_parameters.left_cam;
        //     int u = static_cast<int>((center3D.x * calib.fx) / center3D.z + calib.cx);
        //     int v = static_cast<int>((center3D.y * calib.fy) / center3D.z + calib.cy);
        //     cv::circle(cv_image, cv::Point(u, v), 5, cv::Scalar(0, 0, 255), -1);
        // }

        // // Compute orientation using PCA and print Euler angles
        // cv::Point3d centroid;
        // cv::Vec3d euler;

        // if (computePCAOrientation(validPoints3D, centroid, euler))
        // {
        //     applyTemporalSmoothing(centroid, euler);
        //     {
        //         std::lock_guard<std::mutex> lock(pose_mutex);
        //         latest_position = centroid;
        //         latest_orientation = euler;
        //     }
        //     std::cout << "6D Pose: T[" << centroid.x << ", " << centroid.y << ", " << centroid.z << "] ";
        //     std::cout << "R[" << euler[0] * 180 / CV_PI << ", " << euler[1] * 180 / CV_PI << ", " << euler[2] * 180 / CV_PI << "]" << std::endl;
        //     // std::cout << "Accepted: contour with area=" << areaPixels << std::endl;
        // }

        // // std::cout << "Accepted: contour with area=" << areaPixels << " " << std::endl;

        // 3D pose
        cv::Point3d centroid;
        cv::Vec3d euler;
        cv::Mat eigen;
        bool has_pose = computePCAOrientation(validPoints3D, centroid, euler, &eigen);
        if (has_pose)
        {
            // Smooth pose using recent history
            applyTemporalSmoothing(centroid, euler);

            // Update global pose for TCP/IP use
            {
                std::lock_guard<std::mutex> lock(pose_mutex);
                latest_position = centroid;
                latest_orientation = euler;
            }

            // Project and draw 3D axes (from PCA eigenvectors)
            auto calib = zed.getCameraInformation().camera_configuration.calibration_parameters.left_cam;
            cv::Point center2D(
                static_cast<int>((centroid.x * calib.fx) / centroid.z + calib.cx),
                static_cast<int>((centroid.y * calib.fy) / centroid.z + calib.cy));
            double axis_len = 0.2; // meters (length of axis to draw)

            for (int i = 0; i < 3; ++i)
            {
                cv::Point3d axis_end = centroid + axis_len * cv::Point3d(eigen.at<double>(i, 0), eigen.at<double>(i, 1), eigen.at<double>(i, 2));
                cv::Point end2D(
                    static_cast<int>((axis_end.x * calib.fx) / axis_end.z + calib.cx),
                    static_cast<int>((axis_end.y * calib.fy) / axis_end.z + calib.cy));
                cv::Scalar color = (i == 0) ? cv::Scalar(0, 0, 255) : (i == 1) ? cv::Scalar(0, 255, 0)
                                                                               : cv::Scalar(255, 0, 0); // X=red, Y=green, Z=blue
                cv::line(display_image, center2D, end2D, color, 2);
            }

            // Draw pose text
            char text[128];
            snprintf(text, sizeof(text),
                     "T[%.2f,%.2f,%.2f] R[%.1f,%.1f,%.1f]",
                     centroid.x, centroid.y, centroid.z,
                     euler[0] * 180 / CV_PI, euler[1] * 180 / CV_PI, euler[2] * 180 / CV_PI);
            cv::putText(display_image, text, bbox.tl() + cv::Point(0, -10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 2);
        }
    }
}
