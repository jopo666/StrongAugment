import numpy
from strong_augment import EXAMPLE_IMAGE, StrongAugment


def test_repeat_agumentation():
    trnsf = StrongAugment()
    for __ in range(128):
        trnsf(EXAMPLE_IMAGE)
    augmented = numpy.array(trnsf(EXAMPLE_IMAGE))
    repeated = numpy.array(trnsf.repeat(EXAMPLE_IMAGE))
    assert numpy.equal(augmented, repeated).all()
    trnsf = StrongAugment(max_affine=0)
    augmented = numpy.array(trnsf(EXAMPLE_IMAGE))
    repeated = numpy.array(trnsf.repeat(EXAMPLE_IMAGE))
    assert numpy.equal(augmented, repeated).all()
