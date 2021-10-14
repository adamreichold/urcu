fn main() {
    println!("cargo:rustc-link-lib=urcu-memb");
    println!("cargo:rustc-cfg=atomic_ptr")
}
