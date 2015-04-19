/**
 * DesKeyTest
 * 
 * A test class for DesKey
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.tests
{
	import com.hurlant.crypto.symmetric.DESKey;
	import com.hurlant.util.Hex;
	import flash.utils.ByteArray;
	
	public class DESKeyTest extends TestCase
	{
		public function DESKeyTest(h:ITestHarness)
		{
			super(h, "DESKey Test");
			runTest(testECB,"DES ECB Test Vectors");
			h.endTestCase();
		}
		
		/**
		 * Test vectors mostly grabbed from
		 * http://csrc.nist.gov/publications/nistpubs/800-17/800-17.pdf
		 * (Appendix A and B)
		 * incomplete.
		 */
		public function testECB():void {
			var keys:Array = [
			"3b3898371520f75e", // grabbed from the output of some js implementation out there
			"10316E028C8F3B4A", // appendix A vector
			"0101010101010101", // appendix B Table 1, round 0
			"0101010101010101", // round 1
			"0101010101010101", // 2
			"0101010101010101", 
			"0101010101010101",
			"0101010101010101",
			"0101010101010101",
			"0101010101010101",
			"0101010101010101", // round 8
			"8001010101010101", // app B, tbl 2, round 0
			"4001010101010101",
			"2001010101010101",
			"1001010101010101",
			"0801010101010101",
			"0401010101010101",
			"0201010101010101",
			"0180010101010101",
			"0140010101010101", // round 8
			 ];
			var pts:Array = [
			"0000000000000000", // js
			"0000000000000000", // App A
			"8000000000000000", // App B, tbl 1, rnd0
			"4000000000000000",
			"2000000000000000",
			"1000000000000000",
			"0800000000000000", // rnd 4
			"0400000000000000",
			"0200000000000000",
			"0100000000000000",
			"0080000000000000", // round 8
			"0000000000000000", // App B, tbl2, rnd0
			"0000000000000000",
			"0000000000000000",
			"0000000000000000",
			"0000000000000000",
			"0000000000000000",
			"0000000000000000",
			"0000000000000000",
			"0000000000000000", // rnd 8
			 ];
			var cts:Array = [
			"83A1E814889253E0", // js
			"82DCBAFBDEAB6602", // App A
			"95F8A5E5DD31D900", // App b, tbl 1, rnd 0
			"DD7F121CA5015619",
			"2E8653104F3834EA",
			"4BD388FF6CD81D4F",
			"20B9E767B2FB1456",
			"55579380D77138EF",
			"6CC5DEFAAF04512F",
			"0D9F279BA5D87260",
			"D9031B0271BD5A0A", // rnd 8
			"95A8D72813DAA94D", // App B, tbl 2, rnd 0
			"0EEC1487DD8C26D5",
			"7AD16FFB79C45926",
			"D3746294CA6A6CF3",
			"809F5F873C1FD761",
			"C02FAFFEC989D1FC",
			"4615AA1D33E72F10",
			"2055123350C00858",
			"DF3B99D6577397C8", // rnd 8
			 ];
			
			for (var i:uint=0;i<keys.length;i++) {
				var key:ByteArray = Hex.toArray(keys[i]);
				var pt:ByteArray = Hex.toArray(pts[i]);
				var des:DESKey = new DESKey(key);
				des.encrypt(pt);
				var out:String = Hex.fromArray(pt).toUpperCase();
				assert("comparing "+cts[i]+" to "+out, cts[i]==out);
				// now go back to plaintext
				des.decrypt(pt);
				out = Hex.fromArray(pt).toUpperCase();
				assert("comparing "+pts[i]+" to "+out, pts[i]==out);
			}
		}
	}
}