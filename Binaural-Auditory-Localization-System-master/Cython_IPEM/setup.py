from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

setup(
 name = "mycodecpy",
 version = "1.0",
 escription = "mycodecpy",
 ext_modules = cythonize([
  Extension("mycodecpy", ["mycodecpy.pyx", "cpu.c", "cpupitch.c",
                          "decimation.c", "ecebank.c", "filterbank.c",
                          "Hcmbank.c", "command.c", "filenames.c",
                          "pario.c", "sigio.c", "IPEMAuditoryModel.c", "Audimod.c", "AudiProg.c"]),
  ]),
)

