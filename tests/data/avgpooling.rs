pub const AVGPOOLING_BATCH_SIZE: usize = 1;
pub const AVGPOOLING_INPUT_W: usize = 22;
pub const AVGPOOLING_INPUT_H: usize = 12;
pub const AVGPOOLING_INPUT_C: usize = 20;
pub const AVGPOOLING_FILTER_W: usize = 6;
pub const AVGPOOLING_FILTER_H: usize = 5;
pub const AVGPOOLING_STRIDE_W: usize = 9;
pub const AVGPOOLING_STRIDE_H: usize = 5;
// const AVGPOOLING_PAD SAME
pub const AVGPOOLING_ACTIVATION_MAX: i8 = 127;
pub const AVGPOOLING_ACTIVATION_MIN: i8 = -128;
pub const AVGPOOLING_OUTPUT_C: usize = 20;
pub const AVGPOOLING_OUTPUT_W: usize = 3;
pub const AVGPOOLING_OUTPUT_H: usize = 3;
pub const AVGPOOLING_PADDING_H: usize = 1;
pub const AVGPOOLING_PADDING_W: usize = 1;

pub const AVGPOOLING_INPUT_TENSOR: [i8; 5280] = [
    88, -115, 20, 33, -35, 14, 16, 37, 65, 81, -8, 127, -71, 6, 26, -67, -28, 8, -99, -118, -94,
    -96, 58, -95, 2, 15, -19, 77, 77, 101, -51, -109, 9, 75, 11, 78, -108, -79, -94, -22, 77, 15,
    -40, 34, -52, -45, -104, -77, -99, 7, 18, 87, 46, 99, -56, -120, -81, -72, -93, -123, -66, -57,
    95, 107, 1, 15, -116, -48, 81, 35, -98, 39, 115, -67, 48, -97, 44, -58, -94, -116, 22, 105,
    -76, 45, 53, -121, -54, 53, 72, -63, -115, 78, 69, 93, -16, -20, -101, -51, -92, 72, 103, 31,
    25, 61, -56, 56, -62, 123, 30, 73, -59, 87, -126, 30, 34, -5, 115, 11, 15, -51, 8, -85, -90,
    80, -92, 82, 56, 109, 42, 108, 104, -14, 70, 88, -122, 26, 43, 61, -40, 44, -13, 30, 59, -114,
    40, -29, -51, 99, 113, 115, 44, -124, -35, -97, -81, 12, -125, -54, -27, -69, 34, -111, -19,
    -112, 30, -9, -18, -106, -20, -48, -85, -25, -116, 66, -19, -88, 116, -126, 114, -20, 49, 64,
    109, -36, 115, -3, 122, -68, 34, -118, -119, -12, -76, -30, -14, 58, 60, 111, 95, 55, -75, -57,
    -20, 15, 62, -104, -36, 114, 121, 59, -55, 7, 47, -111, -11, 9, 33, 100, 101, -58, 47, -41, 28,
    -9, -46, 124, 87, -36, -34, 67, 109, -73, -110, 71, 52, -89, 121, 23, 85, -25, 110, 95, 80,
    -20, 78, 42, 45, 79, 87, 35, 103, 76, -96, -61, -101, 81, 20, 59, -18, -111, -107, 73, 121,
    112, 113, 33, 49, 65, -62, -35, -119, -28, 99, -9, -27, 92, 28, -34, -14, 110, 94, -33, 89, -3,
    14, -43, 43, 106, -28, -48, -99, -18, 87, -10, 55, 18, 75, 75, 54, 36, -55, -38, 84, 91, 100,
    26, 35, 46, 72, 86, -111, -55, 87, 65, 90, 59, -61, 57, -6, 121, 35, -43, 41, -69, -10, 78,
    -51, 110, -58, -53, 98, 39, 123, 61, -12, 125, 29, -124, -123, 48, 97, -38, 6, -9, 25, 124,
    118, -7, -26, 77, 40, -73, -108, -28, 9, 29, -58, 45, -22, 48, -33, 31, 53, 55, 13, -62, -120,
    94, -61, 74, 43, -31, 15, -93, 36, -46, -18, 104, -53, 124, -23, 107, -29, 38, -34, -11, -6,
    20, 6, 78, -64, -94, -54, -109, 114, -64, 85, 15, -75, -60, 26, 29, 0, 94, -122, -101, 2, 23,
    14, 125, -100, 69, 41, 9, -64, 107, 82, 97, -124, -7, 99, -75, -109, -25, 26, 85, 74, -54, -32,
    -112, -120, -26, -123, 121, 21, -29, -42, 103, 32, -7, -110, -95, -77, -74, 1, 77, 69, 1, 18,
    -10, -7, -14, 112, -16, -51, -109, -91, -32, 59, 25, -71, -83, 27, 52, -69, -71, -34, -42, 101,
    -37, -6, -101, -4, -104, 63, -17, 76, -5, 126, 52, 65, -103, 85, 72, 49, -95, -72, -13, 37, 12,
    13, -70, 39, -112, 107, 65, -118, -127, 37, 98, -125, -120, 61, 113, -100, -79, 104, -51, -108,
    -32, -36, -13, -33, -46, -42, 73, 72, -7, 105, -104, 79, -88, -126, 10, -6, 19, 15, -59, -65,
    25, -66, 43, 12, 118, 38, 80, -89, 62, -80, 120, -9, -113, 100, 117, -122, 62, 126, -100, 58,
    10, 94, 101, -56, -37, -11, -70, 27, 63, 34, 64, 124, -125, 66, 48, 55, 62, 86, -116, 109, 18,
    -123, -57, 38, 5, 13, -78, -44, 1, 106, -76, -96, 70, -38, -81, -51, 65, 7, 86, 50, -49, -78,
    -87, 21, -83, 80, 88, -44, 66, 100, -31, -12, 63, -101, 23, -97, -53, 56, 98, -78, 75, -5, -70,
    18, -65, -67, -65, 69, -87, 14, 5, 83, -38, 98, 42, 22, -8, 28, -9, 39, -51, 1, 73, 70, 46,
    -38, -107, -37, 37, -80, -49, 57, -117, 10, -15, 62, 109, 12, -122, 105, 91, -92, 19, -56, 57,
    -76, -61, -60, 11, 95, -63, -88, 104, -20, 58, -124, 122, 63, -96, 72, 43, 99, 35, -27, 40,
    -64, 64, 85, 59, -47, 115, 12, -38, 46, -25, -55, 94, -25, 46, -117, 66, 60, 43, -62, 92, 98,
    -108, 4, 57, -36, -19, -67, 6, -107, -75, -17, -69, -92, 33, -87, -8, 43, -71, 100, -76, -70,
    79, 125, -110, -51, 52, 28, 30, -50, -127, 95, 12, -49, -41, -65, -70, -48, -10, 120, 39, -125,
    -107, 33, -30, 96, 3, 119, 93, -2, 40, 68, -40, 13, -104, -91, -87, -52, -48, 70, 94, -86, 96,
    45, -16, -53, -13, 65, 41, -34, -63, -111, 116, -18, 16, 66, -123, 56, -49, 42, 125, 118, -77,
    122, 112, 38, 30, -123, -114, -64, -63, 69, -104, 124, -30, 107, 94, -45, -114, 81, -87, 60,
    -2, 32, -12, 51, 119, -115, 113, 43, 56, -121, 52, 18, 78, 68, 115, 47, 54, 13, -49, -121, 88,
    51, -121, 43, 70, -20, 74, 101, -112, 123, -7, 13, -98, 1, 21, 117, 3, -116, 35, 32, -46, -100,
    -110, 54, 29, -94, -91, 82, -126, 101, 72, -123, 31, -85, 29, -22, 10, 44, -88, 90, -48, -19,
    -96, -20, -108, -94, 4, 50, -78, -94, 9, 48, 120, 73, -100, 100, -102, 70, 73, -116, 93, 5,
    -88, -86, 19, -66, 35, 13, 73, -113, -53, -121, -123, 123, -19, 85, -19, 50, 83, 27, -37, 86,
    55, 108, -100, 1, 94, 69, -109, 62, 81, 42, -78, -63, -34, -54, -106, 1, -125, -32, 2, -61,
    -15, 88, -53, 22, 8, 126, -53, 29, 102, -12, 104, -39, -104, -111, 0, -16, -58, 99, 62, 114, 4,
    31, -51, -105, 106, 86, 106, -79, 80, 108, -120, -29, -116, 74, -64, -90, 37, -49, -44, -67, 0,
    -105, -109, -40, -110, 14, -127, 7, -68, 123, -102, 72, 22, 84, -107, -100, -19, 109, -65, -6,
    108, -122, -85, 34, -22, -68, -119, -30, -7, -90, 1, 115, -71, 70, -63, -23, 125, -38, -96,
    -39, 97, -69, 85, -3, -90, 22, -9, -18, 23, -117, 62, -32, 109, -51, -50, 8, 56, 108, -95, 75,
    -58, -26, -112, -4, -114, 88, -49, 93, -27, -15, 15, 0, -92, 83, -61, 46, -93, -19, 106, -98,
    104, -2, 40, -7, -102, 16, 86, -125, 57, 125, -74, -19, -37, -74, -78, -8, 14, -14, 9, 12, -73,
    -1, -50, 14, -27, -53, 49, 104, 75, -58, -50, -108, -5, 26, -61, 58, -93, -68, -1, -78, 106,
    -116, 55, -99, 125, 34, 5, -53, 17, 62, -16, 117, 64, 1, -97, 103, -44, 11, 112, -96, -25, 59,
    -122, -21, 21, -106, -51, 53, -43, -82, -103, -103, 95, 20, -52, -108, 57, 124, 116, 125, -46,
    75, -2, 117, 119, 23, 3, 87, 100, -91, 115, -116, -58, -55, -122, 80, 24, -110, -17, -110, -47,
    -107, 25, 70, 70, 17, -34, 48, 126, 27, 28, -42, 3, 1, -45, -12, 58, -100, 67, 111, -101, 112,
    16, 123, 125, -120, 118, -90, 88, -100, -127, -84, 54, -28, 55, -16, -78, -4, -113, 95, -48,
    -125, -120, -112, -128, -121, -118, -108, 27, -88, -122, 91, 125, -113, 82, -33, -75, -23, -91,
    71, 48, 12, 97, 101, 65, -95, 90, -83, -118, -71, -35, -83, 50, -22, -21, 10, 49, -109, 90,
    -105, -77, -95, -70, -43, -88, -82, -6, -80, -101, 46, -6, -39, 20, 93, 103, -6, -65, 97, -16,
    102, 59, -36, -116, -12, -55, 62, 88, 11, 73, 27, -113, -22, -120, -6, -59, -83, -87, 70, -127,
    -109, 56, 99, 126, 8, 118, -125, 70, -8, -39, 18, 121, -113, 60, 36, 114, 119, -68, 98, 91,
    -20, -120, 78, 70, -7, -66, 104, -6, -32, 89, -46, -15, -42, -82, 85, -54, 28, -56, -16, 75,
    127, -38, -77, -105, -33, -81, -92, 104, -47, -42, -85, 120, -84, -113, 55, -39, -100, 6, 31,
    -31, 7, -86, -67, -82, -125, 47, -77, 44, -57, 41, 79, -37, -63, -97, 71, 69, 56, -50, -95,
    -73, 111, -97, -7, -67, -55, -117, -69, 34, 80, 23, -104, 41, 7, 94, -90, -33, 14, 102, 29,
    -19, 88, -41, 92, -128, -60, 55, 126, -33, -24, -98, 10, 105, 13, 112, 121, -121, 121, 107,
    -80, -62, 62, 127, -117, -29, -127, 18, -47, -52, 71, -103, 31, -119, -126, -65, -69, 29, -19,
    -11, -119, 34, 88, -1, 1, 33, 79, 16, 22, 30, 58, -30, 42, 47, 63, 41, -85, -90, 83, -117, -98,
    -123, 120, -27, 22, -100, 93, -75, -62, 4, 19, -122, 83, 6, 36, 102, 95, 16, -117, -102, -53,
    40, -23, 13, 40, -102, -109, 17, -87, 91, -27, 99, -65, -118, 106, 35, 62, 39, 1, -110, -98,
    -117, 84, 15, -66, 87, -8, -120, 27, -68, -69, 52, -104, -62, -23, 55, -88, -33, -115, -99, 10,
    -19, -85, -80, -99, -16, 4, 112, -123, 88, 115, 86, 2, 74, 9, 117, -121, -108, -88, -37, 49,
    61, -127, -41, -103, 118, 55, -114, 36, 36, -55, -58, -59, -63, 52, -48, -15, -94, 108, 50,
    121, 52, -55, -59, 83, -12, 46, 107, 24, -8, -85, -73, 24, 53, -73, -27, 32, 15, 113, 9, -57,
    -58, 69, 109, 120, -37, -89, 47, -111, -19, 83, -61, -36, 92, -14, -28, -20, -19, -82, 16, -92,
    -43, 108, 109, -128, -45, -13, 84, -117, 40, 106, -122, 74, 75, -107, 56, -81, 86, -53, -107,
    21, -18, 53, -43, -64, -31, -27, -125, -93, -109, -64, 31, -31, -102, 121, -70, 10, -127, -5,
    -124, -58, -47, -12, 11, -89, -34, -108, -109, -49, -36, 69, 108, 82, -47, 35, -94, -29, -7,
    -80, -29, 92, 81, -122, -38, -77, -8, -91, -96, 109, -95, 94, 91, -27, -60, 115, -1, 57, 110,
    22, -50, -1, -69, -30, -103, 83, -63, -124, 81, 42, 16, -4, 116, 57, 92, -46, -14, 42, 111,
    -83, 17, -22, 97, 30, -107, 70, -54, -70, -28, 96, 88, 103, 45, 119, 101, 99, 15, 69, 15, -17,
    88, 63, -95, 80, 48, 111, -105, -83, 118, -41, 64, 104, 115, -30, -111, 126, -125, 82, 50, -90,
    94, 53, -58, 110, 28, 70, -88, 6, -46, -121, 30, -59, -73, -14, -7, -52, -14, 119, 0, -22, 57,
    -16, -9, -53, -99, 69, -58, -117, 111, 61, 108, 36, -50, -36, -57, 44, 57, 127, -21, -104, -37,
    -1, -33, 78, -124, -70, 66, 94, 73, 35, 113, 100, 42, 31, -62, 60, -55, -4, 77, -99, 33, 3, 33,
    -25, 68, -75, 44, -115, 0, 108, 49, -19, -55, 96, 21, 69, 124, -24, 73, -124, 59, 1, 75, -99,
    42, -102, -118, 57, 121, -112, 93, -70, -82, -115, 21, 112, -101, 83, -16, -89, 60, 95, -98,
    -29, -109, 8, -95, 4, -34, 91, 13, 69, -108, 27, 55, -93, 15, -51, 112, -20, 112, -14, -59, 81,
    -67, 47, 115, -17, 57, 125, -94, -23, -78, 33, 57, 114, -98, -22, 5, -109, -84, 90, -104, -122,
    60, -107, -77, 59, 86, 31, 103, 99, 12, 122, 102, 119, 42, 27, -124, -19, 105, -114, -37, 114,
    31, -34, -9, 76, -80, 6, 110, -4, -5, 7, 87, 23, 27, 39, -111, 7, 9, 41, 113, 76, 77, -14, 63,
    116, 127, -116, 50, -16, 48, 61, -90, -118, 7, 39, 115, -64, -15, 44, 33, 119, -76, -10, -14,
    112, -81, -76, -84, 90, -48, -57, -73, 13, -28, -101, 10, -127, 58, -94, -103, 52, -126, 100,
    -47, -127, -64, 96, -13, 86, 124, -54, -125, 21, 120, -68, -123, 16, 69, 49, 45, -88, -97, -37,
    115, -26, 100, -103, -23, -96, 57, 67, 53, 44, -51, 53, -61, 12, -35, -104, -96, 106, 45, 4, 9,
    60, -5, 104, 51, -46, -125, -7, 47, 55, 104, 70, -58, 105, 94, -50, -57, -75, -116, -25, -112,
    24, 87, -62, 24, -93, -74, -48, 113, -94, 23, 85, 56, -110, -98, -40, 66, 98, -118, 85, 41,
    -55, 58, -2, -79, -21, 22, -38, -87, -27, 45, 78, -17, -89, -107, 78, 108, 74, -21, 107, 87,
    -48, -126, -14, 24, -56, 33, 116, -123, 121, 67, -64, -49, 97, 64, 84, 79, 83, -54, 93, 77, 51,
    -48, -62, -59, 109, -83, -78, 20, 107, -115, 112, -38, -92, 104, -94, 103, -123, -127, 75, -73,
    -42, -55, 21, 96, 108, -67, 22, -51, 84, 47, -98, 63, -51, 53, 98, 101, -109, -50, -11, 66, 20,
    -109, 108, 44, -112, 52, -49, 28, 97, 98, 119, -27, 34, 100, 1, -9, -52, 49, -70, 124, -1, -22,
    45, -53, -9, 11, 118, 113, -7, -2, -55, 84, 104, 40, -63, -71, 90, 118, 12, 0, 126, 89, -8,
    -79, 64, 25, 76, -4, -101, 85, -3, -59, -2, 98, 57, 58, 24, -17, 38, 77, 65, -90, -11, 59, 83,
    -1, 75, -53, 94, -90, 89, -122, -26, 73, 61, -87, 30, -83, 44, 77, -21, -38, 66, 122, 72, 91,
    77, 43, -109, -19, -62, 24, -28, -6, -122, -4, -18, 115, -52, 50, 69, 29, 99, -25, -52, -8,
    -79, 22, 106, -120, -41, 62, 50, 122, -1, 96, 67, -12, -112, -21, -82, -79, -45, -88, -122, 99,
    -3, 97, -56, -84, 41, -17, -123, 44, -7, 73, 39, 40, -54, -81, -82, 106, 119, -54, 96, -20,
    -125, -91, 64, -126, -26, 59, 54, -119, -53, -48, -102, 116, 3, -20, -91, 91, -102, -105, -93,
    57, 60, -32, -55, 81, -74, 59, 13, 12, -110, 32, -34, -122, -24, -102, -63, -97, 108, -62, 120,
    -13, -74, -106, -29, -108, 60, -94, 84, 60, 22, 81, 80, 16, 50, 31, -12, 44, 43, -87, 110,
    -110, 115, 21, 2, 38, -67, -75, -104, -26, -51, 28, -25, -97, -68, -105, -40, 31, 7, -107, 121,
    -125, 49, -7, 80, -124, -43, -92, -121, 61, 86, 98, -115, 20, -66, 57, -64, -86, -105, 14,
    -114, 80, 98, 92, 68, 41, 102, -32, -41, 101, -107, 118, -119, -74, -27, -77, -7, 40, 77, 75,
    -54, 90, -83, -19, 60, -123, -110, -79, -93, 96, 52, 76, -112, -49, 63, 96, 31, -126, -49,
    -111, 98, 10, -86, -120, -82, -84, -94, -57, 30, -34, 120, -20, -128, -80, 12, -97, -117, -7,
    -82, 68, -52, 95, 5, 74, -75, -125, 69, 126, -102, -97, 58, 83, -79, -103, 10, 43, -10, -28,
    80, 90, 100, 115, 76, 12, 95, -107, -71, 79, -66, 51, 30, -26, 118, -117, -29, -44, 68, 62,
    -70, -26, -35, -21, -63, -4, -86, -118, 126, 111, -47, 38, -127, -16, -114, 116, 72, -98, -84,
    -46, -110, -98, -79, 88, -101, 72, -109, -116, -79, -63, 72, 3, -92, -104, -68, -105, 92, -83,
    41, 49, -88, -116, 90, 55, 28, -66, 96, -95, -109, -32, -112, -9, -17, 54, 24, 126, -80, 1,
    118, -67, 58, 75, 35, -108, -8, 109, -30, -120, -22, 12, 0, -5, 85, -95, -45, 74, -109, 57,
    122, -29, 74, 119, 105, 124, 12, 8, -18, 24, 4, 75, -81, -78, 44, -26, -28, 123, -3, -87, -78,
    -118, 113, 99, -116, 115, 41, -53, -126, -125, 49, -123, -4, 38, 124, -109, -9, 104, -8, -86,
    -52, 124, 64, -56, 109, 29, -29, -91, 123, 52, -120, 56, -77, -94, 117, -98, 97, -85, -69,
    -104, 15, 10, -55, 8, -26, -6, -25, 100, 41, -70, -71, -47, -51, 4, -84, -21, 89, -48, 108,
    123, -92, -3, 21, -12, 124, -22, -24, 106, -85, -84, 96, 84, 65, 28, -83, 68, 70, 96, 46, -123,
    -36, 25, -103, -104, 95, -121, 32, 81, 49, -8, 91, -120, 47, 48, -70, -80, -23, 11, -13, -128,
    -59, 6, -60, 48, -74, -125, 69, 41, -7, 114, -91, 22, -99, 119, -112, -70, 63, -38, -7, 11,
    -46, -46, 59, 28, 82, 63, 107, -28, 111, 42, -11, -111, -65, -113, -116, 22, -12, 2, -96, 14,
    -90, 49, 15, 90, -84, 32, 92, 8, -75, 73, 45, 49, 79, -83, -69, -79, -93, 72, 90, -47, -92,
    -69, -39, 107, -92, 52, 99, -109, 64, 49, -3, 40, -64, -48, -89, 13, 63, -3, 52, -95, -107, 70,
    -47, 33, -87, -21, -114, -98, 20, -120, -113, 118, 112, 35, 41, -66, 78, -87, 4, -115, -127,
    -90, -59, 71, -34, 5, 105, 95, 62, -81, 51, -96, 60, -12, -126, -96, 27, 61, 36, 121, -86, 19,
    -60, -102, 51, 39, -124, -67, -66, -17, -59, 105, 99, 1, -85, 38, 56, -86, 85, -91, -69, 69,
    -90, 47, -121, -100, -37, -123, -2, 10, 0, -43, 30, 95, 70, 109, 59, 15, -34, -78, 88, -34,
    125, -41, 75, -55, -103, 8, -91, -50, 111, 74, -30, 7, -29, 12, 57, -117, 112, 106, -42, 109,
    -111, -94, 69, -56, 59, -53, -10, 119, -115, 97, 40, -57, 6, 17, -1, 58, 125, 74, -10, -83, 90,
    51, -52, 2, -3, -22, -45, -90, 41, -117, 55, -12, -92, -57, -40, -20, 39, 26, 81, 23, 22, 81,
    29, -104, -22, -9, -117, 39, -51, 79, 46, -114, 91, -7, -125, 0, 37, -35, -35, 4, 25, 62, 10,
    36, 14, -46, -40, 10, -93, -125, 70, -113, 35, -122, -45, 30, -15, -32, -4, -48, -42, -107, 41,
    44, -2, 62, 19, -36, -45, -86, -45, 113, -60, 62, -21, 91, -83, -102, 111, -2, -77, 73, -55,
    18, 6, -67, -4, -41, 88, -71, 93, 54, 36, 99, 96, 108, 74, 80, -114, -85, 93, -23, -103, -8,
    66, 82, 73, -35, -37, 27, -22, -28, 90, -3, -90, 16, 0, 83, 9, 34, -83, -62, 7, 47, -120, -46,
    -29, 121, 17, -21, 117, -50, 60, 89, -83, 31, -18, 96, -5, -102, 70, 96, -128, 53, 11, 65, -36,
    36, 48, 3, -112, -67, 62, -13, -57, 116, -5, 108, 22, -9, 0, -105, 80, -55, -112, 49, 21, 92,
    16, -94, -52, 70, -107, -38, -18, -105, 86, 28, 112, -55, 29, 68, 74, -100, -29, -5, 20, -89,
    -41, -99, -124, 20, -34, 116, -46, 81, 34, 46, 73, 93, 27, 14, -29, 123, 5, -75, -11, -59,
    -118, -42, 79, -8, 60, -99, -73, -88, 63, -86, 123, -42, 97, -66, -91, 61, 46, -21, 20, -75,
    -10, -55, -56, -41, 56, -55, 118, -39, -40, 98, -127, -116, -75, 87, 12, 62, -28, 50, 72, -5,
    119, -90, -53, 50, -105, -96, -65, 127, -95, -67, -117, -58, 115, -117, -93, -35, -121, -33,
    -43, 36, 83, -42, -3, -47, -110, -44, 83, -107, -45, -70, -65, 21, -63, -124, 65, 114, 108, 70,
    -108, 123, -65, 16, -5, -127, 84, -37, -128, 85, -8, 29, -51, 86, 87, -19, 101, 39, -18, -109,
    14, -87, 56, 34, 27, -66, 86, 20, 23, -85, 70, 101, 83, 11, 57, 105, -8, 119, 110, -57, 75, 80,
    -55, -32, 8, 40, -30, 73, -21, -86, 122, -40, 42, -56, 106, -38, 27, -113, 6, 65, -73, 20, -40,
    -101, -84, -62, -120, 57, -73, -44, 37, -46, -64, 108, -1, -92, 45, -18, -39, -28, 27, 56, -72,
    -18, 99, 36, 1, 70, -13, -51, -89, -117, 20, -55, -70, -64, -114, 127, 3, 7, 33, 43, -85, -66,
    -92, 0, 61, -95, -4, 12, -109, -127, 21, -128, 44, 103, -23, 126, 43, -37, -123, -67, 83, 93,
    -25, 107, -20, -26, 50, -73, 25, 66, -37, 107, 93, -122, 106, -103, -49, -52, -103, 52, -83,
    -103, -54, -58, -103, 62, -93, -108, -47, 106, -12, 0, 47, 36, -68, 37, -126, -73, -51, -76,
    35, 107, -112, -33, -88, 94, -118, -18, 6, -82, -3, 98, 3, -17, 116, 15, 125, 18, -29, -114,
    56, -13, 55, 49, -72, 59, 50, -82, 41, -108, 79, 75, 2, -120, -13, 124, 104, 60, 29, 42, -38,
    40, 54, -48, -107, -17, -124, -15, -29, 7, -8, -7, 123, 23, -80, 76, 83, 121, 123, 42, 51, 46,
    81, -109, -34, 3, -109, -72, -109, -107, -104, -39, -48, -111, -17, 106, 47, 127, -39, -68,
    -106, 58, 48, 95, 84, 18, -109, 91, 107, -29, -88, -41, 3, -72, 78, -61, -42, -44, 118, 31, 66,
    28, -19, -53, 119, -40, -15, 46, 102, 94, 36, -39, -102, -83, -14, 33, -62, -80, -53, 14, -2,
    -11, -95, 76, -114, 92, 73, -12, -110, 31, 61, 89, 97, -124, -72, 40, 39, -54, 71, -20, -115,
    21, -122, 51, -65, -127, -60, 58, -20, 20, -69, 63, 6, -5, -83, 69, -120, 31, -120, -95, 98,
    93, 77, 29, 0, -46, -59, 84, 120, 70, 76, -4, 110, 11, -94, 29, -9, -100, 97, -41, 65, 3, 20,
    -67, 115, -95, 8, 113, -40, -22, -97, 19, 23, -102, 75, -61, 119, 82, 88, 112, 22, -93, -28,
    -3, -124, 21, 31, -32, -117, -26, -71, -53, 91, 97, -91, -91, -19, -94, 69, 10, 68, 37, -60,
    99, 89, 68, 76, -28, 50, 46, 73, -57, -106, 63, -97, 105, 97, -117, -98, 104, 62, -9, 97, 27,
    91, -107, -53, -56, -42, -125, -89, -50, -74, 83, 102, 62, -17, 12, -82, 85, -57, -102, 67, 23,
    -112, 15, 10, -90, 86, 88, -119, 77, -28, -122, -32, -103, 111, 21, 122, 116, 125, 42, -1, -83,
    -96, -70, -33, 70, 105, 2, 15, -115, 24, 29, -86, -61, 4, 31, 121, 103, -89, -76, -31, -52, 21,
    36, 109, 27, 72, 83, -42, 78, 61, 56, -94, -112, -42, 54, 56, -126, 4, -92, -72, 101, 27, 81,
    -119, 24, -30, -105, -104, -33, 74, -37, -6, -46, 58, 19, -125, 100, -98, 20, 115, 28, -77,
    -102, 74, 76, -85, -2, 99, 118, -111, -117, 105, 16, 120, 98, -122, 65, 104, -26, -15, 15, -82,
    99, 7, 2, 120, 9, 5, 48, 29, 14, -10, -50, 73, -77, -87, 90, -16, -4, -44, -80, 12, -12, 55,
    -70, -78, -113, -73, 17, 85, -104, -17, -65, 50, 66, -74, -58, 36, -125, 125, 88, -71, 86, -74,
    -23, 115, -72, -39, -87, 24, -112, -64, 84, 17, -11, 4, -74, -63, 113, 37, 117, 9, -96, -94,
    -107, -16, 127, 48, -28, -18, 20, 1, -12, 61, -10, 28, -106, 45, -56, 17, -54, -49, 52, -58,
    -28, -32, -112, 112, -53, 65, 116, -28, 60, -89, 9, -85, -126, -21, -21, -120, -109, -44, -110,
    -81, 63, 78, 37, 61, -127, 33, 63, -81, 64, 84, 61, 13, 49, 5, -44, -53, 55, -5, -43, -80, 87,
    43, -30, 70, -100, -117, 90, -42, -75, -20, 11, -82, 99, 100, -21, 40, -91, -120, -44, 15, -50,
    -49, -3, 24, -30, -68, -82, 76, -53, -54, -44, -33, -9, 112, -44, 91, -103, 70, 1, 81, 96, -15,
    32, -114, 120, -19, 44, -19, 64, -30, -111, 18, -75, -84, 52, 95, -102, -74, 19, 54, 116, -19,
    15, 47, 76, -127, -78, 29, -121, -98, -61, 100, 31, -56, -48, -120, 20, 101, 107, 7, 21, -62,
    106, -111, -42, 39, -66, 125, -57, 62, -50, -12, -65, -29, -30, 23, 55, 74, 1, 109, -53, -113,
    79, -116, 12, 17, -6, 49, -66, 93, 16, 18, 32, -6, -27, 123, -20, 39, -86, 15, -63, -99, 107,
    22, 44, 91, -5, 65, -52, 48, -125, -12, 105, 26, -26, -109, 86, -61, -117, 49, -37, -114, -72,
    79, 15, -42, -101, -99, 105, 119, -101, 59, -40, 64, 31, -95, -70, 89, -49, -119, 86, -33, 43,
    -110, 5, 109, 61, 110, -79, 8, 11, -128, 89, 82, 87, -42, -68, -61, -22, -123, -126, 91, 77,
    -100, -15, -114, -111, 95, 97, 72, 119, 36, -10, 117, 72, -126, -87, 115, 124, 41, -94, -44,
    -59, -101, -84, -58, -52, 33, 50, -50, -35, -27, -99, -9, 80, 62, -117, 54, 114, -110, -39,
    -45, 32, -52, -88, 125, -68, -89, 59, 11, -41, 69, 83, 103, -84, -2, 91, -34, -120, -22, -19,
    17, -82, -27, 72, 9, 110, -4, -119, -20, 80, -15, 11, -77, -72, 6, 61, -126, 64, 100, 71, -31,
    56, -120, -83, -56, -6, -4, 63, -22, -51, -84, -24, 6, 26, -85, -70, 68, -95, -125, -82, -79,
    14, -19, 63, 118, -48, -21, -11, 7, -103, 52, 2, 51, 40, 102, -87, -95, -28, -21, -79, 89, -59,
    -4, -127, 101, -125, -102, -33, -75, -21, 40, -94, -52, 52, -14, -9, -119, -47, 97, 51, -83,
    82, 54, 36, 3, -88, 45, 106, 106, 67, -22, 95, -46, -82, -55, -56, 23, 16, 111, 74, -105, 42,
    46, 58, 46, 67, -43, 5, -108, 124, 43, -59, 33, 113, -102, -83, 93, 115, -36, 127, 78, 121,
    109, 95, 97, 60, -117, -31, -45, -79, 72, 3, 58, -56, 47, 91, -54, -38, 3, 2, -72, 124, -32,
    -26, -11, 77, 42, 65, -88, -126, 46, 61, 107, 111, 23, -99, -84, -43, -109, -72, 43, -123, 45,
    20, 64, -90, -119, -36, 91, 104, -6, -11, 121, -18, -88, 30, -19, -60, -66, 113, 96, -123, -18,
    -43, -113, 87, 21, 1, 53, -92, -14, 53, 25, 64, 92, -81, -65, -127, 101, -56, -29, 13, 34, 93,
    -107, -29, -8, -103, 35, -32, 0, -26, 111, 34, 1, 0, -58, -108, 58, -95, -9, -57, -70, 88, -64,
    4, 47, -102, -47, -23, -18, -5, 57, 56, -79, -113, -66, -15, 90, -52, -124, 0, -111, -67, -76,
    -120, -95, 8, -19, 31, 3, -29, -26, 7, 0, 16, 104, -125, 22, -71, 88, -120, -78, 28, -123, -65,
    9, 27, 78, -79, 86, -56, -85, -21, 78, 99, 21, -59, -121, 59, 47, 8, -13, 122, -66, -104, -35,
    65, -93, -65, 72, -46, -24, 53, 63, 120, 41, -75, 92, 67, 0, 43, 28, 113, -38, -81, -37, -4,
    16, -102, -44, -25, -45, -32, -94, 8, -105, -18, -66, -78, -13, -103, 91, 6, 110, 84, 67, 89,
    -86, -117, 14, 100, -49, 125, 109, 59, 124, 104, 115, 58, -50, 21, -103, 84, 84, 89, -52, 118,
    -102, 91, -65, -121, -92, 120, -9, -96, 23, -16, -26, -114, -113, 124, 89, -19, -126, 13, -122,
    27, -11, -20, 79, 111, -111, 119, -85, 82, 73, 102, 102, 41, -83, 78, 10, 68, 33, 19, 81, -108,
    -22, 80, 57, 39, -46, -88, 34, 51, 69, 18, 68, 107, -25, -66, 30, 46, -124, 14, -35, -5, -122,
    -46, 97, -90, -118, 115, -74, 76, -3, -111, 40, 93, -25, -27, 43, -42, -6, -8, 88, 71, -63,
    -96, -120, 43, 48, 13, 9, 13, 47, 57, 126, -35, -63, 75, -102, 4, -62, 115, 69, -22, -126, 121,
    -47, 106, 108, 24, -67, -83, 126, 69, -27, -100, 66, 114, 88, -100, 35, 17, 86, 117, -74, 8,
    15, -2, -45, 122, 1, -9, 3, -36, 92, -43, -79, 48, 88, 19, 80, 103, 42, -11, -29, -46, 35, -88,
    -71, 105, 49, 31, -63, 19, 91, -125, -58, 35, 124, -120, -19, -13, 108, -36, -36, 78, 64, -96,
    -123, 112, -104, -3, -50, 84, -104, -107, 46, 9, -68, 4, -67, -38, 0, 76, -117, 68, 20, 45, 47,
    -78, -53, 80, -62, -106, -108, 61, -92, 55, -70, -95, -5, 52, 99, -119, 41, 59, 40, -72, -117,
    66, -30, 92, -31, 81, 103, 32, 34, -36, 44, -78, 105, -64, 4, 115, 55, 4, -65, 88, -52, -45,
    24, -98, -70, -35, 99, -108, -27, 86, -36, -104, -23, 1, -13, -16, -103, -65, 65, 11, 110, 84,
    -58, -4, 77, 45, 12, 47, -4, -62, -42, 19, -29, 0, 50, -33, 20, 56, 39, 47, 73, 107, 116, -117,
    88, 117, 119, 92, -127, 107, -59, 15, 44, -53, -6, 94, 114, 119, -109, -80, 55, -46, -87, 106,
    94, 107, 5, 50, 47, 40, -42, 10, 10, -34, -69, -92, 55, 122, 46, 121, 97, -95, -43, -13, 103,
    56, 49, -49, 50, 78, -32, -36, 27, 77, 64, 33, 23, 53, -5, 125, 96, -76, 98, 77, 108, 17, -123,
    -33, -55, -92, 52, -85, 74, 99, 76, -6, -35, -44, -111, -98, 25, -76, 66, -61, -7, 119, 126,
    -51, -17, 51, 25, -28, -26, 80, 70, 23, 33, 114, -78, -107, -122, 110, -8, 108, -118, -95, 23,
    -83, 2, 125, 41, -48, 94, -11, 64, -42, -4, 108, -19, -47, 57, 71, -81, -62, -6, 77, 59, -100,
    -31, -123, -71, -117, 83, 117, -62, -19, -100, 66, -40, -60, -60, 80, -61, 101, -94, -20, 25,
    119, 125, -13, 114, 96, 95, 66, -83, 9, -86, -58, 96, -73, 91, 49, 103, 64, -15, 63, 100, 83,
    -88, 109, 32, 69, -46, 27, 21, 68, 20, -14, -121, -63, -126, 54, 10, -27, 73, 88, -101, -34,
    -91, 121, 94, -78, 5, 102, -75, -119, -10, 40, -89, 24, -120, -102, 94, -100, 95, -64, -33,
    -47, -124, 119, 116, 68, -14, 17, -26, 46, 60, -50, 0, 108, -100, 39, -116, 30, -29, -93, -126,
    124, 116, 73, -80, 92, 91, 56, 34, 52, -80, 14, -51, 2, 72, -12, 65, 127, -77, -7, 110, 105,
    63, -23, 43, -11, -99, -52, 84, 103, -86, 0, 33, -110, -46, 102, -114, 82, 104, -21, -113, 96,
    -19, 29, -61, -77, -94, 71, -31, -64, -18, 102, 44, 122, 73, -77, 107, -68, 85, 75, 29, 5, -7,
    -37, -125, 17, -12, -72, 30, 80, 87, -13, 2, -111, 38, -15, -118, -88, 63, -107, -17, -90, 83,
    -119, 19, 51, 87, 70, -128, 2, -128, -61, -32, 29, 29, 80, -98, 118, 126, 47, 111, -70, 46,
    101, -11, -52, 106, 5, 37, -25, -21, 125, 13, -69, -58, -72, 4, -51, -52, -91, -81, -104, 61,
    -64, -56, 105, 72, 0, -49, 57, 18, 49, 87, -75, -117, 102, 112, 108, 62, 64, 75, -17, -94, 98,
    40, 100, -73, -50, 112, 19, 125, -39, 27, 98, -71, 36, -21, 93, 82, -78, 69, -32, -54, -73, 24,
    87, 117, 111, -52, 2, 98, 3, -46, 15, -19, -69, -117, -101, 24, 50, -66, -17, -98, 123, -114,
    51, 66, 5, 57, -44, 64, 26, -65, 15, -29, -77, 40, -127, -37, 8, -47, 102, 117, 11, 69, -4,
    117, -78, -35, -32, -106, 100, -21, 25, -54, 95, 14, -117, -6, 65, 84, -111, 18, 19, -56, 70,
    -58, 82, 106, 99, -125, 95, 113, 35, 86, 48, -103, -116, -1, 58, -126, -75, 114, 107, -99,
    -105, -119, -105, 86, -101, -99, -60, 114, 28, -114, -62, 45, -29, 123, 64, 38, -11, -81, 60,
    -113, 87, -98, 29, -7, 14, 122, 47, -71, -117, -121, -128, 58, 1, 17, 47, -87, 83, 75, 91, -68,
    113, 115, -113, 4, 29, 60, -69, 57, 25, -74, -124, -78, -91, 86, -86, 109, 86, -20, -112, -15,
    44, -82, 75, 35, -58, -14, -53, 12, -64, 121, -16, -106, -48, 105, -78, -64, 67, -114, -109,
    -46, 112, -101, -107, 125, 52, -3, -65, -61, -119, -57, 113, -81, -105, -63, -91, -96, -117,
    -69, 41, 88, -94, -103, -119, 97, -40, 11, -34, -69, 18, 23, -107, -106, 21, 120, -65, -125,
    -108, 118, -25, -19, -4, 16, -124, 127, -117, -71, -6, -122, 20, -65, -72, 119, -88, 11, 6, -3,
    7, -71, -123, -78, 45, 109, -33, -26, -12, 92, -37, -98, -48, 70, 0, 90, 95, -82, 89, 94, 64,
    -94, 107, 15, -87, -111, -27, 61, 79, 68, 39, -43, -55, 20, -5, -96, 57, -85, 45, -107, 52, 61,
    44, 41, 43, -20, 64, -44, -107, 5, -111, 120, -113, 16, -47, 16, -35, -64, -9, 107, -16, 125,
    -60,
];

pub const AVGPOOLING_OUTPUT: [i8; 180] = [
    -31, -24, 15, 11, -6, -15, 0, -20, 7, -2, -17, -19, 20, 0, -4, -23, 6, -12, -12, -40, 4, 13,
    -10, 2, 15, 17, 10, 1, -9, 10, -14, -30, -4, -7, -21, 4, 24, 0, 4, -5, -27, 5, 2, 24, 6, 6, 9,
    7, 2, 2, 33, -6, -26, -13, 22, -4, 16, -14, -5, -1, -50, 7, 26, -3, -3, -12, 25, 3, 15, 8, -9,
    -14, -12, -10, -33, -9, -4, -7, 5, -22, 2, -8, -8, -12, -5, 21, 4, -2, -15, -21, -3, -8, 5,
    -16, -15, 3, -30, -13, 15, -7, -4, 1, 13, -8, 14, 1, 23, 3, 9, 0, 16, -9, -15, 18, -6, -28, 11,
    -5, -7, 1, -25, -8, 2, 17, 10, 24, -48, -11, -2, -8, 11, 30, -10, 5, 19, -5, 18, -21, -3, 10,
    -22, 13, 1, 4, 3, -9, 7, 14, 4, 22, 34, -4, -1, -39, 10, 12, 2, -11, -4, 22, 30, 13, 5, 14, -5,
    -6, -14, -16, -22, -9, -40, 9, 1, 15, 8, 15, 2, 17, -14, 27,
];
