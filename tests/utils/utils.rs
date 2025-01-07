pub fn validate(act: *const i8, ref_data: *const i8, size: usize) -> bool {
    let mut test_passed = true;
    let mut count = 0;
    let mut total = 0;

    for i in 0..size {
        total += 1;
        let act_data = unsafe { *act.offset(i as _) };
        let ref_data = unsafe { *ref_data.offset(i as _) };
        if act_data != ref_data {
            defmt::error!("ERROR at pos {}: Act: {} Ref: {}", i, act_data, ref_data);
            count += 1;
            test_passed = false;
        }
    }

    if !test_passed {
        defmt::error!("{} of {} failed", count, total);
    }

    test_passed
}