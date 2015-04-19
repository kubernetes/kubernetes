/**
 * TripleDESKeyTest
 * 
 * A test class for TripleDESKey
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.tests
{
	import com.hurlant.crypto.symmetric.TripleDESKey;
	import com.hurlant.util.Hex;
	import flash.utils.ByteArray;
	import com.hurlant.crypto.symmetric.ICipher;
	import com.hurlant.crypto.symmetric.ECBMode;
	
	public class TripleDESKeyTest extends TestCase
	{
		public function TripleDESKeyTest(h:ITestHarness)
		{
			super(h, "Triped Des Test");
			runTest(testECB,"Triple DES ECB Test Vectors");
			h.endTestCase();
		}
		
		/**
		 * Lots of vectors at http://csrc.nist.gov/publications/nistpubs/800-20/800-20.pdf
		 * XXX move them in here.
		 */
		public function testECB():void {
			var keys:Array = [
			"010101010101010101010101010101010101010101010101",
			"dd24b3aafcc69278d650dad234956b01e371384619492ac4",
			];
			var pts:Array = [
			"8000000000000000",
			"F36B21045A030303",
			];
			var cts:Array = [
			"95F8A5E5DD31D900",
			"E823A43DEEA4D0A4",
			];
			
			for (var i:uint=0;i<keys.length;i++) {
				var key:ByteArray = Hex.toArray(keys[i]);
				var pt:ByteArray = Hex.toArray(pts[i]);
				var ede:TripleDESKey = new TripleDESKey(key);
				ede.encrypt(pt);
				var out:String = Hex.fromArray(pt).toUpperCase();
				assert("comparing "+cts[i]+" to "+out, cts[i]==out);
				// now go back to plaintext
				ede.decrypt(pt);
				out = Hex.fromArray(pt).toUpperCase();
				assert("comparing "+pts[i]+" to "+out, pts[i]==out);
			}
		}
		
	}
}