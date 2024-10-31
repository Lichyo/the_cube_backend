import where_the_magic_happened


def picture_in_result_out(RGBImage):
    resultString,probabilities = where_the_magic_happened.imageHandPosePredict(RGBImage)
    return resultString,probabilities
