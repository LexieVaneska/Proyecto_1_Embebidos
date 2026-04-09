require recipes-core/images/core-image-minimal.bb

SUMMARY = "Image with myapp + X11 display support"

IMAGE_INSTALL += " \
    myapp \
    opencv \
    gstreamer1.0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-plugins-base-meta \
    gstreamer1.0-libav \
    xserver-xorg \
    xserver-xorg-extension-glx \
    xf86-video-vesa \
    xinit \
    xterm \
    gtk+3 \
    libx11 \
    v4l-utils \
    openssh \
    bash \
"

# Handy extras for debugging inside VirtualBox
IMAGE_INSTALL += "openssh bash procps"

IMAGE_FEATURES:append = " x11 ssh-server-openssh"
