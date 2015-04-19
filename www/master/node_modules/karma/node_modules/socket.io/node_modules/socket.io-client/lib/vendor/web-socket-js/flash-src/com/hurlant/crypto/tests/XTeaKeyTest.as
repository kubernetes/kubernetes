/**
 * XTeaKeyTest
 * 
 * A test class for XTeaKey
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.tests
{
	import com.hurlant.crypto.prng.Random;
	import com.hurlant.crypto.symmetric.ECBMode;
	import com.hurlant.crypto.symmetric.XTeaKey;
	import com.hurlant.util.Hex;
	
	import flash.utils.ByteArray;
	import flash.utils.getTimer;
	
	public class XTeaKeyTest extends TestCase
	{
		public function XTeaKeyTest(h:ITestHarness) {
			super(h, "XTeaKey Test");
			runTest(testGetBlockSize, "XTea Block Size");
			runTest(testVectors, "XTea Test Vectors");
			
			h.endTestCase();
		}
		
		public function testGetBlockSize():void {
			var tea:XTeaKey = new XTeaKey(Hex.toArray("deadbabecafebeefdeadbabecafebeef"));
			assert("tea blocksize", tea.getBlockSize()==8);
		}
		
		public function testVectors():void {
			// blah.
			// can't find working test vectors.
			// algorithms should not get published without vectors :(
			var keys:Array=[
			"00000000000000000000000000000000",
			"2b02056806144976775d0e266c287843"];
			var pts:Array=[
			"0000000000000000",
			"74657374206d652e"];
			var cts:Array=[
			"2dc7e8d3695b0538",
			"7909582138198783"];
			// self-fullfilling vectors.
			// oh well, at least I can decrypt what I produce. :(
			
			for (var i:uint=0;i<keys.length;i++) {
				var key:ByteArray = Hex.toArray(keys[i]);
				var pt:ByteArray = Hex.toArray(pts[i]);
				var tea:XTeaKey = new XTeaKey(key);
				tea.encrypt(pt);
				var out:String = Hex.fromArray(pt);
				assert("comparing "+cts[i]+" to "+out, cts[i]==out);
				// now go back to plaintext.
				pt.position=0;
				tea.decrypt(pt);
				out = Hex.fromArray(pt);
				assert("comparing "+pts[i]+" to "+out, pts[i]==out);
			}
		}

	}
}