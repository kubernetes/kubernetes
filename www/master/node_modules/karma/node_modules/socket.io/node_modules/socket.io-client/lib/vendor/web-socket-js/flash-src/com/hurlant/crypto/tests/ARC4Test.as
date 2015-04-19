/**
 * ARC4Test
 * 
 * A test class for ARC4
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.tests
{
	import com.hurlant.util.Hex;
	import flash.utils.ByteArray;
	import com.hurlant.crypto.prng.ARC4;
	
	public class ARC4Test extends TestCase
	{
		public function ARC4Test(h:ITestHarness)
		{
			super(h, "ARC4 Test");
			runTest(testLameVectors,"ARC4 Test Vectors");
			h.endTestCase();
		}
		
		/**
		 * Sad test vectors pilfered from
		 * http://en.wikipedia.org/wiki/RC4
		 */
		public function testLameVectors():void {
			var keys:Array = [
			Hex.fromString("Key"),
			Hex.fromString("Wiki"),
			Hex.fromString("Secret")];
			var pts:Array = [
			Hex.fromString("Plaintext"),
			Hex.fromString("pedia"),
			Hex.fromString("Attack at dawn")];
			var cts:Array = [
			"BBF316E8D940AF0AD3",
			"1021BF0420",
			"45A01F645FC35B383552544B9BF5"];
			
			for (var i:uint=0;i<keys.length;i++) {
				var key:ByteArray = Hex.toArray(keys[i]);
				var pt:ByteArray = Hex.toArray(pts[i]);
				var rc4:ARC4 = new ARC4(key);
				rc4.encrypt(pt);
				var out:String = Hex.fromArray(pt).toUpperCase();
				assert("comparing "+cts[i]+" to "+out, cts[i]==out);
				// now go back to plaintext
				rc4.init(key);
				rc4.decrypt(pt);
				out = Hex.fromArray(pt);
				assert("comparing "+pts[i]+" to "+out, pts[i]==out);
			}
		}
		
	}
}