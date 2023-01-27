# -*- coding: utf-8 -*-
import unittest
from IPEM import *

class TestIPEMCases(unittest.TestCase):
    def test_output(self):
        # 呼叫程式
        self.myIPEM = IPEM('violin.wav', '', 'violin.txt', '')
        self.my_IPEM = self.myIPEM.ipem()

        # 讀取程式output
        self.cython_filename = 'violin.txt'
        self.cythonfile = open(self.cython_filename, 'r')
        self.cython = self.cythonfile.read()

        # 讀取cygwin output
        self.cygwin_filename = 'cygwin_result.txt'
        self.cygwinfile = open(self.cygwin_filename, 'r')
        self.cygwin = self.cygwinfile.read()

        # 判斷
        self.flag = False
        if self.cython == self.cygwin:
            self.flag = True

        self.assertEqual(self.flag, True)

if __name__ == '__main__':
    unittest.main()
