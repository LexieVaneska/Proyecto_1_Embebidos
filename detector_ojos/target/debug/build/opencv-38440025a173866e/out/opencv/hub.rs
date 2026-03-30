#[cfg(ocvrs_has_module_core)]
include!(concat!(env!("OUT_DIR"), "/opencv/core.rs"));
#[cfg(ocvrs_has_module_highgui)]
include!(concat!(env!("OUT_DIR"), "/opencv/highgui.rs"));
#[cfg(ocvrs_has_module_imgproc)]
include!(concat!(env!("OUT_DIR"), "/opencv/imgproc.rs"));
#[cfg(ocvrs_has_module_objdetect)]
include!(concat!(env!("OUT_DIR"), "/opencv/objdetect.rs"));
#[cfg(ocvrs_has_module_videoio)]
include!(concat!(env!("OUT_DIR"), "/opencv/videoio.rs"));
pub mod types {
	include!(concat!(env!("OUT_DIR"), "/opencv/types.rs"));
}
#[doc(hidden)]
pub mod sys {
	include!(concat!(env!("OUT_DIR"), "/opencv/sys.rs"));
}
pub mod hub_prelude {
	#[cfg(ocvrs_has_module_core)]
	pub use super::core::prelude::*;
	#[cfg(ocvrs_has_module_highgui)]
	pub use super::highgui::prelude::*;
	#[cfg(ocvrs_has_module_imgproc)]
	pub use super::imgproc::prelude::*;
	#[cfg(ocvrs_has_module_objdetect)]
	pub use super::objdetect::prelude::*;
	#[cfg(ocvrs_has_module_videoio)]
	pub use super::videoio::prelude::*;
}
