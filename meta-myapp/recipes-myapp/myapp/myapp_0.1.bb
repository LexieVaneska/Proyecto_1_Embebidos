SUMMARY = "Rust + OpenCV embedded project"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

SRC_URI = " \
    git://github.com/LexieVaneska/Proyecto_1_Embebidos.git;protocol=https;branch=main \
    file://lbfmodel.yaml \
"
PV = "1.0+git${SRCPV}"
SRCREV = "${AUTOREV}"

S = "${WORKDIR}/git/detector_ojos"

inherit cargo_bin

do_fetch[network] = "1"
do_compile[network] = "1"

export CLANG_PATH = "${STAGING_BINDIR_NATIVE}/clang"
export LIBCLANG_PATH = "${STAGING_DIR_NATIVE}/usr/lib"
export LLVM_CONFIG_PATH = "${STAGING_BINDIR_NATIVE}/llvm-config"

export CPLUS_INCLUDE_PATH = "${STAGING_DIR_TARGET}/usr/include/c++/11.5.0:${STAGING_DIR_TARGET}/usr/include/x86_64-oe-linux/c++/11.5.0:${STAGING_DIR_TARGET}/usr/include"
export C_INCLUDE_PATH = "${STAGING_DIR_TARGET}/usr/include"

do_compile:prepend() {
    GENERATOR_RS="${WORKDIR}/cargo_home/registry/src/index.crates.io-1949cf8c6b5b557f/opencv-binding-generator-0.90.2/src/generator.rs"
    if [ -f "$GENERATOR_RS" ]; then
        sed -i 's/args\.push("-std=c++14"\.into());/args.push("-std=c++14".into());\n\t\t\targs.push("-D_GLIBCXX20_DEPRECATED(x)=".into());\n\t\t\targs.push("-D_GLIBCXX_DEPRECATED(x)=".into());/' "$GENERATOR_RS"
    fi
    TYPE_TRAITS="${WORKDIR}/recipe-sysroot/usr/include/c++/11.5.0/type_traits"
    if [ -f "$TYPE_TRAITS" ]; then
        chmod +w "$TYPE_TRAITS"
        sed -i '/^    _GLIBCXX20_DEPRECATED("use is_standard_layout && is_trivial instead")$/{N;s/    _GLIBCXX20_DEPRECATED("use is_standard_layout && is_trivial instead")\n    is_pod/    is_pod/}' "$TYPE_TRAITS"
    fi
    rm -rf ${WORKDIR}/target/release/deps/libopencv_binding_generator* || true
}

DEPENDS += " \
    opencv \
    gtk+3 \
    libx11 \
    xrandr \
    clang \
    clang-native \
"

RDEPENDS:${PN} += " \
    opencv \
    gtk+3 \
    libx11 \
    xrandr \
    xauth \
"

do_install() {
    install -d ${D}${bindir}
    install -m 0755 ${CARGO_BINDIR}/detector_ojos ${D}${bindir}/detector_ojos

    # Install model files
    install -d ${D}${datadir}/myapp/models
    install -m 0644 ${WORKDIR}/lbfmodel.yaml ${D}${datadir}/myapp/models/lbfmodel.yaml

    # Install input videos
    install -d ${D}${datadir}/myapp/videos
    for video in ${S}/videos/*.mp4; do
        [ -f "$video" ] && install -m 0644 "$video" ${D}${datadir}/myapp/videos/
    done

    # Create output directory
    install -d ${D}${datadir}/myapp/videos_salida
}

FILES:${PN} += "${datadir}/myapp"
