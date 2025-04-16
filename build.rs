use std::{
    env,
    path::{Path, PathBuf},
};
use which::which;
use embuild::bindgen::types::Formatter;

const CMSIS_NN_URL: &str = "https://github.com/ARM-software/CMSIS-NN.git";
const CMSIS_VERSION: &str = "v7.0.0";

fn main() {
    let target = env::var("TARGET").unwrap();

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    let out_path = PathBuf::from("target");

    let _cmsis_nn_repo = get_repo(&CMSIS_NN_URL, &out_path);

    let arm_toolchain = env::var("ARM_TOOLCHAIN_PATH").unwrap_or_else(|_| {
        let gcc_path = which("arm-none-eabi-gcc").unwrap();
        gcc_path
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string()
    });

    let cmsis_nn_dir = out_path.join("CMSIS-NN").join(CMSIS_VERSION);

    let toolchain_file = manifest_dir.join("cmake/toolchain/arm-none-eabi-gcc.cmake");

    let target_flag = match target.as_str() {
        "thumbv7em-none-eabihf" => "cortex-m4",
        "thumbv7m-none-eabi" => "cortex-m3",
        _ => unimplemented!(),
    };
    let lib = embuild::cmake::Config::new(&cmsis_nn_dir)
        .define("CMAKE_TOOLCHAIN_FILE", &toolchain_file)
        .define("TARGET_CPU", target_flag)
        .build_target("cmsis-nn")
        .build()
        .join("build");

    println!("cargo:rustc-link-search={}", lib.display());

    println!("cargo:rustc-link-lib=static=cmsis-nn");

    let cmsis_nn_include_dir = cmsis_nn_dir.join("Include");
    let arm_toolchain_include_dir = PathBuf::from(arm_toolchain).join("include");

    let bindings = embuild::bindgen::Factory::new()
        .with_linker("arm-none-eabi-gcc")
        .builder()
        .unwrap()
        .header("wrapper.h")
        .derive_default(true)
        .generate_comments(true)
        .formatter(Formatter::Rustfmt)
        .clang_arg(format!("-I{}", cmsis_nn_include_dir.display()))
        // .clang_arg("-nostdinc")
        .clang_arg(format!("-I{}", arm_toolchain_include_dir.display()))
        .use_core()
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    println!("cargo::rerun-if-changed=schema/schema.fbs");
    flatc_rust::run(flatc_rust::Args {
        inputs: &[Path::new("schema/schema.fbs")],
        out_dir: &Path::new("target/flatbuffers/"),
        ..Default::default()
    })
    .expect("flatc");
}

fn get_repo(url: &str, path: &PathBuf) -> embuild::git::Repository {
    let git_ref = embuild::git::Ref::parse(CMSIS_VERSION);
    let remote_repo = embuild::git::sdk::RemoteSdk {
        repo_url: None,
        git_ref,
    };
    let clone_options = embuild::git::CloneOptions::new();

    let repo = remote_repo
        .open_or_clone(&path, clone_options, url, "CMSIS-NN")
        .unwrap();

    repo
}
