// Parte 1B_2: Red Super Resolución en Video
// Nombre: Bryan Avila
// Carrera: Computación
// Materia: Visión por Computadora
// Fecha: 2025-07-06

// Librerias por utilizar
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <chrono>

// Definición de espacios de nombres
using namespace std;
using namespace cv;
using namespace std::chrono;

// ONNX Runtime environment 
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SuperRes");

// Función que aplica super resolución 
Mat superResolucionFrame(const Mat& input_img, Ort::Session& session, Ort::MemoryInfo& memory_info,
                         const vector<const char*>& input_names, const vector<const char*>& output_names) {
    int h = input_img.rows;
    int w = input_img.cols;

    // Convertir BGR a RGB y normalizar
    Mat img_rgb;
    cvtColor(input_img, img_rgb, COLOR_BGR2RGB);
    img_rgb.convertTo(img_rgb, CV_32F, 1.0f / 255.0f);

    // Separar canales
    vector<Mat> rgb_channels(3);
    split(img_rgb, rgb_channels);

    // Preparar input tensor [1, 3, h, w]
    vector<float> input_tensor_values(1 * 3 * h * w);
    vector<int64_t> input_shape = {1, 3, h, w};

    for (int c = 0; c < 3; ++c) {
        memcpy(&input_tensor_values[c * h * w], rgb_channels[c].data, h * w * sizeof(float));
    }

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size()
    );

    // Ejecutar inferencia
    auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                      input_names.data(), &input_tensor, 1,
                                      output_names.data(), 1);

    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto out_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    int out_c = (int)out_shape[1];
    int out_h = (int)out_shape[2];
    int out_w = (int)out_shape[3];

    vector<Mat> output_channels(out_c);
    for (int c = 0; c < out_c; ++c) {
        output_channels[c] = Mat(out_h, out_w, CV_32F, output_data + c * out_h * out_w).clone();
    }

    Mat output_rgb;
    merge(output_channels, output_rgb);

    // De-normalizar y convertir a 8-bit
    output_rgb = output_rgb * 255.0f;
    output_rgb.convertTo(output_rgb, CV_8U);

    // Convertir RGB a BGR para mostrar
    Mat output_bgr;
    cvtColor(output_rgb, output_bgr, COLOR_RGB2BGR);

    return output_bgr;
}

int main() {
    // Ruta del modelo ONNX
    string model_path = "/home/bryan/Documentos/segundoInterciclo/TrabajoU4PartB2/models/RealESRGAN_x2.onnx";
    // Ruta del video de entrada
    string video_path = "/home/bryan/Documentos/segundoInterciclo/TrabajoU4PartB2/video5.mp4";

    // Variable para seleccionar si usar GPU (true) o CPU (false)
    bool usarGPU = true;

    // Configuración de ONNX Runtime
    Ort::SessionOptions session_options;

    // Mejoras de rendimiento generales
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.EnableMemPattern();
    session_options.EnableCpuMemArena();

    // Para usar solo la CPU comentar toda esta condición
    if (usarGPU) {
        OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
        if (status != nullptr) {
            cerr << "Error al habilitar CUDA en ONNX Runtime" << endl;
            return -1;
        }
        cout << ">>> Ejecutando con GPU" << endl;
    }

     cout << ">>> Ejecutando con CPU" << endl; // Descomentar para usar CPU

    // Crear la sesión
    Ort::Session session(env, model_path.c_str(), session_options);
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Obtener nombres de entrada y salida
    vector<const char*> input_names;
    vector<const char*> output_names;
    vector<Ort::AllocatedStringPtr> input_name_ptrs;
    vector<Ort::AllocatedStringPtr> output_name_ptrs;

    size_t num_inputs = session.GetInputCount();
    for (size_t i = 0; i < num_inputs; ++i) {
        input_name_ptrs.emplace_back(session.GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions()));
        input_names.push_back(input_name_ptrs.back().get());
    }

    size_t num_outputs = session.GetOutputCount();
    for (size_t i = 0; i < num_outputs; ++i) {
        output_name_ptrs.emplace_back(session.GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions()));
        output_names.push_back(output_name_ptrs.back().get());
    }

    // Abrir video
    VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        cerr << "No se pudo abrir el video: " << video_path << endl;
        return -1;
    }

    int fps = static_cast<int>(cap.get(CAP_PROP_FPS));
    int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));

    // Reducir tamaño para evitar uso excesivo memoria
    float scale_factor = 0.5f;
    int resized_width = static_cast<int>(frame_width * scale_factor);
    int resized_height = static_cast<int>(frame_height * scale_factor);

    // Ventana de visualización
    namedWindow("Video Super Resolución: RealESRGAN", WINDOW_AUTOSIZE);

    // Medición de tiempo para FPS
    auto start_time = high_resolution_clock::now();
    int frame_count = 0;

    Mat frame;
    while (cap.read(frame)) {
        // Redimensionar frame de entrada
        resize(frame, frame, Size(resized_width, resized_height));

        // Aplicar super resolución al frame
        Mat sr_frame = superResolucionFrame(frame, session, memory_info, input_names, output_names);

        // Contador de frames y cálculo de FPS
        frame_count++;
        auto now = high_resolution_clock::now();
        double elapsed = duration_cast<duration<double>>(now - start_time).count();
        double real_fps = frame_count / elapsed;

        // Dibujar FPS en pantalla
        string fps_text = "FPS: " + to_string(static_cast<int>(real_fps));
        putText(sr_frame, fps_text, Point(20, 40), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);

        // Mostrar frame
        imshow("Video Super Resolución: RealESRGAN", sr_frame);

        // Salir si se presiona ESC
        if (waitKey(1) == 27) {
            break;
        }
    }

    cap.release();
    destroyAllWindows();

    return 0;
}
