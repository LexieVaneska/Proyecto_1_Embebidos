# Detector de Estado Ocular en Video

Aplicación desarrollada en Rust con OpenCV para analizar un video cuadro por cuadro, detectar el rostro principal, estimar landmarks faciales y clasificar si cada ojo está abierto o cerrado. Como salida, el sistema muestra el video original y el procesado en pantalla, y además genera un video anotado en `videos_salida/video_procesado.mp4`.

## Integrantes

- Alexandra Alfaro Elizondo
- Jose Pablo Esquetini Fallas
- Kendall Madrigal Campos
- Taller de Sistemas Embebidos
- Instituto Tecnologico de Costa Rica

## Objetivo del proyecto

Construir una aplicacion multimedia en Rust con soporte de OpenCV e integrarla dentro de una imagen Linux personalizada generada con Yocto Project para arquitectura `x86_64`.

## Flujo general de la aplicacion

1. Lee un video de entrada desde la carpeta `videos/`.
2. Detecta el rostro principal en cada cuadro con Haar Cascade.
3. Estima landmarks faciales usando `Facemark LBF`.
4. Analiza ambos ojos mediante metricas geometricas y visuales.
5. Dibuja etiquetas, regiones de interes y una grafica historica del EAR.
6. Guarda el resultado en `videos_salida/video_procesado.mp4`.

## Estructura del repositorio

```text
detector_ojos/
├── Cargo.toml
├── models/
│   └── lbfmodel.yaml
├── src/
│   ├── analysis.rs
│   ├── config.rs
│   ├── io.rs
│   ├── main.rs
│   ├── overlay.rs
│   └── processing.rs
├── videos/
│   ├── input_video1.mp4
│   ├── input_video3.mp4
│   ├── input_video4.mp4
│   └── input_video5.mp4
└── videos_salida/
    └── video_procesado.mp4
```

## Requisitos previos

### Software utilizado

- Ubuntu 24.04.4 LTS como host
- Rust 1.75 o superior
- Cargo
- OpenCV 4
- Clang/LLVM
- VirtualBox 7.2.6
- Yocto Project / Poky `kirkstone`

### Dependencias de host para compilar con Yocto

```bash
sudo apt update && sudo apt upgrade -y

sudo apt install -y \
  gawk wget git diffstat unzip texinfo gcc build-essential \
  chrpath socat cpio python3 python3-pip python3-pexpect \
  xz-utils debianutils iputils-ping python3-git python3-jinja2 \
  libegl1-mesa libsdl1.2-dev pylint xterm python3-subunit \
  mesa-common-dev zstd liblz4-tool file locales libacl1 \
  clang llvm

sudo locale-gen en_US.UTF-8
```

### Dependencias para ejecutar localmente la aplicacion

Ademas de Rust, el sistema necesita OpenCV con los modulos usados por el proyecto:

- `face`
- `highgui`
- `imgproc`
- `objdetect`
- `videoio`

En Ubuntu suele ser necesario instalar al menos:

```bash
sudo apt install -y clang llvm pkg-config libopencv-dev
```

## Tutorial de flujo de uso

### 1. Clonar el repositorio

```bash
git clone https://github.com/LexieVaneska/Proyecto_1_Embebidos.git
cd Proyecto_1_Embebidos/detector_ojos
```

### 2. Verificar archivos requeridos

Antes de ejecutar, confirme que existan:

- El video de entrada dentro de `videos/`
- El modelo `models/lbfmodel.yaml`
- El clasificador Haar en `/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml`

### 3. Revisar la configuracion de entrada y salida

Las rutas principales estan definidas en [src/config.rs](/home/alex/Desktop/Tarea-Embebidos/Proyecto_1_Embebidos/detector_ojos/src/config.rs).

Valores importantes:

- `INPUT_VIDEO_PATH`: video que se va a procesar
- `OUTPUT_DIRECTORY`: carpeta donde se guarda la salida
- `OUTPUT_VIDEO_PATH`: archivo final generado
- `FACEMARK_MODEL_PATH`: ruta del modelo de landmarks
- `FACE_CASCADE_PATH`: ruta del clasificador de rostro

Ejemplo actual:

```rust
pub const INPUT_VIDEO_PATH: &str = "videos/input_video5.mp4";
pub const OUTPUT_VIDEO_PATH: &str = "videos_salida/video_procesado.mp4";
pub const FACEMARK_MODEL_PATH: &str = "models/lbfmodel.yaml";
```

Si desea probar otro video, cambie `INPUT_VIDEO_PATH` y vuelva a compilar.

### 4. Ejecutar la aplicacion localmente

```bash
cargo run
```

Durante la ejecucion:

- Se abre una ventana con el video original.
- Se abre una ventana con el video procesado.
- Se escribe el archivo de salida en `videos_salida/video_procesado.mp4`.
- Puede cerrar el proceso presionando la tecla `q`.

### 5. Revisar el resultado

Al finalizar, el programa reporta:

- Ruta del video de entrada
- Resolucion detectada
- FPS del video
- Ruta del video de salida
- Cantidad de cuadros procesados

El resultado final queda almacenado en:

```bash
videos_salida/video_procesado.mp4
```

## Casos de uso evaluados

- Deteccion de parpadeo mientras la persona sonrie
- Prueba de codigo morse con parpadeos
- Analisis con lentes para medir robustez del sistema

## Integracion con Yocto Project

Esta seccion resume el flujo para empaquetar la aplicacion dentro de una imagen Linux personalizada.

### 1. Preparar carpeta de trabajo

```bash
mkdir -p ~/yocto
cd ~/yocto
```

### 2. Clonar capas necesarias

```bash
git clone -b kirkstone git://git.yoctoproject.org/poky.git
git clone -b kirkstone https://github.com/openembedded/meta-openembedded.git
git clone -b master https://github.com/rust-embedded/meta-rust-bin.git
git clone -b kirkstone https://github.com/kraj/meta-clang.git
git clone -b meta-myapp https://github.com/LexieVaneska/Proyecto_1_Embebidos.git meta-myapp
```

### 3. Inicializar el entorno de build

```bash
cd ~/yocto
source poky/oe-init-build-env build
```

### 4. Configurar `local.conf`

Ajuste `build/conf/local.conf` con una base similar a esta:

```conf
MACHINE = "genericx86-64"

IMAGE_FSTYPES += "wic.vmdk"

DISTRO_FEATURES:append = " x11 opengl"

RUST_VERSION = "1.78.0"

PACKAGECONFIG:append:pn-opencv = " gtk"
PACKAGECONFIG:append:pn-opencv = " dnn"
EXTRA_OECMAKE:append:pn-opencv = " -DBUILD_opencv_objdetect=ON -DBUILD_opencv_face=ON"

LICENSE_FLAGS_ACCEPTED += "commercial"

BB_NO_NETWORK = "0"
SRCREV_pn-myapp = "${AUTOREV}"

EXTRA_IMAGE_FEATURES += "debug-tweaks"
EXTRA_IMAGE_FEATURES += "ssh-server-openssh"

BB_NUMBER_THREADS = "8"
PARALLEL_MAKE = "-j 8"
```

### 5. Configurar `bblayers.conf`

Ajuste `build/conf/bblayers.conf` para incluir las capas usadas por el proyecto:

```conf
BBLAYERS ?= " \
  /home/usuario/yocto/poky/meta \
  /home/usuario/yocto/poky/meta-poky \
  /home/usuario/yocto/poky/meta-yocto-bsp \
  /home/usuario/yocto/meta-openembedded/meta-oe \
  /home/usuario/yocto/meta-openembedded/meta-python \
  /home/usuario/yocto/meta-openembedded/meta-multimedia \
  /home/usuario/yocto/meta-rust-bin \
  /home/usuario/yocto/meta-myapp \
  /home/usuario/yocto/meta-clang \
  "
```

Reemplace `/home/usuario/yocto` por su ruta real.

### 6. Compilar la imagen

```bash
bitbake myapp-image
```

### 7. Ubicar la imagen generada

```bash
ls ~/yocto/build/tmp-glibc/deploy/images/genericx86-64/*.wic.vmdk
```

La imagen util para VirtualBox es la terminada en `.wic.vmdk`.

## Instalacion y prueba en VirtualBox

### 1. Crear la maquina virtual

Configure una VM `x86_64` y adjunte la imagen `.wic.vmdk` como disco SATA.

### 2. Ajustes recomendados

- Activar `UEFI`
- Asignar memoria RAM suficiente para entorno grafico
- Mantener controlador de almacenamiento tipo SATA

### 3. Ejecutar el sistema

Una vez iniciada la VM, ejecutar:

```bash
root

export DISPLAY=:0
xinit &

cd /usr/share/myapp
detector_ojos
```

Nota: segun la configuracion de la imagen, el usuario `root` puede iniciar sin contrasena.

## Problemas comunes

### No abre el video de entrada

Revise que el archivo configurado en `INPUT_VIDEO_PATH` exista realmente dentro de `videos/`.

### No encuentra el clasificador Haar

Verifique la ruta de `FACE_CASCADE_PATH` en [src/config.rs](/home/alex/Desktop/Tarea-Embebidos/Proyecto_1_Embebidos/detector_ojos/src/config.rs#L6) y confirme que OpenCV instalo `haarcascade_frontalface_default.xml` en esa ubicacion.

### No encuentra el modelo de landmarks

Revise que exista `models/lbfmodel.yaml`.

### La aplicacion compila pero no muestra ventanas

Confirme que OpenCV fue compilado con soporte `highgui` y que el sistema tiene entorno grafico disponible.

## Referencias consultadas

- Rust: https://www.rust-lang.org/
- Cargo: https://doc.rust-lang.org/cargo/
- Crate `opencv`: https://crates.io/crates/opencv
- Docs del crate `opencv`: https://docs.rs/opencv/
- OpenCV: https://opencv.org/
- Documentacion OpenCV: https://docs.opencv.org/
- MediaPipe Solutions Guide: https://ai.google.dev/edge/mediapipe/solutions/guide?hl=es-419
