/**
 * TLSPRFTest
 * 
 * A test class for TLFPRF
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.tests
{
	import flash.utils.ByteArray;
	import com.hurlant.crypto.prng.TLSPRF;
	import com.hurlant.util.Hex;
	
	public class TLSPRFTest extends TestCase
	{
		public function TLSPRFTest(h:ITestHarness) {
			super(h, "TLS-PRF Testing");
			runTest(testVector, "TLF-PRF Test Vector");
			h.endTestCase()
		}
		
		/**
		 * Test Vector as defined in
		 * http://www.imc.org/ietf-tls/mail-archive/msg01589.html
		 */
		private function testVector():void {
			var secret:ByteArray = new ByteArray;
			for (var i:uint=0;i<48;i++) {
				secret[i]= 0xab;
			}
			var label:String = "PRF Testvector";
			var seed:ByteArray = new ByteArray;
			for (i=0;i<64;i++) {
				seed[i] = 0xcd;
			}
			var prf:TLSPRF = new TLSPRF(secret, label, seed);
			var out:ByteArray = new ByteArray;
			prf.nextBytes(out, 104);
			var expected:String = "D3 D4 D1 E3 49 B5 D5 15 04 46 66 D5 1D E3 2B AB" + 
					"25 8C B5 21 B6 B0 53 46 3E 35 48 32 FD 97 67 54" + 
					"44 3B CF 9A 29 65 19 BC 28 9A BC BC 11 87 E4 EB" + 
					"D3 1E 60 23 53 77 6C 40 8A AF B7 4C BC 85 EF F6" + 
					"92 55 F9 78 8F AA 18 4C BB 95 7A 98 19 D8 4A 5D" + 
					"7E B0 06 EB 45 9D 3A E8 DE 98 10 45 4B 8B 2D 8F" + 
					"1A FB C6 55 A8 C9 A0 13";
			var expect:String = Hex.fromArray(Hex.toArray(expected));
			assert("out == expected", Hex.fromArray(out)==expect);
		}
	}
}