use git2::Repository;
use std::{
    env,
    path::{Path, PathBuf},
};
use which::which;

const CMSIS_NN_URL: &str = "https://github.com/ARM-software/CMSIS-NN.git";

fn switch_branch(repo: &Repository, branch_name: &str) {
    let (object, reference) = repo.revparse_ext(branch_name).unwrap();
    repo.checkout_tree(&object, None).unwrap();
    match reference {
        Some(gref) => repo.set_head(gref.name().unwrap()),
        None => repo.set_head_detached(object.id()),
    }
    .unwrap();
}

fn main() {
    let target = env::var("TARGET").unwrap();
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

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cmsis_nn_dir = out_path.join("CMSIS-NN");

    let cmsis_nn_repo = get_repo(&CMSIS_NN_URL, &cmsis_nn_dir);

    switch_branch(&cmsis_nn_repo, "v7.0.0");

    let toolchain_file = manifest_dir.join("cmake/toolchain/arm-none-eabi-gcc.cmake");

    let target_flag = match target.as_str() {
        "thumbv7em-none-eabihf" => "cortex-m4",
        _ => unimplemented!(),
    };

    let lib = cmake::Config::new(&cmsis_nn_dir)
        .define("CMAKE_TOOLCHAIN_FILE", &toolchain_file)
        .define("TARGET_CPU", target_flag)
        .build_target("cmsis-nn")
        .build()
        .join("build");

    println!("cargo:rustc-link-search={}", lib.display());

    println!("cargo:rustc-link-lib=static=cmsis-nn");

    let cmsis_nn_include_dir = cmsis_nn_dir.join("Include");
    let arm_toolchain_include_dir = PathBuf::from(arm_toolchain).join("include");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .derive_default(true)
        .generate_comments(true)
        .formatter(bindgen::Formatter::Rustfmt)
        .clang_arg(format!("-I{}", cmsis_nn_include_dir.display()))
        // .clang_arg("-nostdinc")
        .clang_arg(format!("-I{}", arm_toolchain_include_dir.display()))
        .use_core()
        .generate()
        .expect("Unable to generate bindings");

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

fn get_repo(url: &str, path: &PathBuf) -> Repository {
    let repo = match Repository::open(&path) {
        Ok(repo) => repo,
        Err(e) if e.code() == git2::ErrorCode::NotFound => Repository::clone(url, &path).unwrap(),
        Err(e) => panic!("Failed to open: {}", e),
    };
    repo
}
