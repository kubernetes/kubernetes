/**
 * OFBModeTest
 * 
 * A test class for OFBMode
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.tests
{
	import flash.utils.ByteArray;
	import com.hurlant.util.Hex;
	import com.hurlant.crypto.symmetric.OFBMode;
	import com.hurlant.crypto.symmetric.AESKey;
	import com.hurlant.crypto.symmetric.NullPad;
	
	public class OFBModeTest extends TestCase
	{
		public function OFBModeTest(h:ITestHarness)
		{
			super(h, "OFBMode Test");
			runTest(testOFB_AES128,"OFB AES-128 Test Vectors");
			runTest(testOFB_AES192,"OFB AES-192 Test Vectors");
			runTest(testOFB_AES256,"OFB AES-256 Test Vectors");
			h.endTestCase();
		}
		
		/**
		 * Nist Vectors: http://csrc.nist.gov/publications/nistpubs/800-38a/sp800-38a.pdf
		 * Section F.4.1 and below.
		 */
		public function testOFB_AES128():void {
			var key:ByteArray = Hex.toArray("2b7e151628aed2a6abf7158809cf4f3c");
			var pt:ByteArray = Hex.toArray(
				"6bc1bee22e409f96e93d7e117393172a" +
				"ae2d8a571e03ac9c9eb76fac45af8e51" + 
				"30c81c46a35ce411e5fbc1191a0a52ef" + 
				"f69f2445df4f9b17ad2b417be66c3710");
			var ct:ByteArray = Hex.toArray(
				"3b3fd92eb72dad20333449f8e83cfb4a" +
				"7789508d16918f03f53c52dac54ed825" + 
				"9740051e9c5fecf64344f7a82260edcc" + 
				"304c6528f659c77866a510d9c1d6ae5e");
			var ofb:OFBMode = new OFBMode(new AESKey(key), new NullPad);
			ofb.IV = Hex.toArray("000102030405060708090a0b0c0d0e0f");
			var src:ByteArray = new ByteArray;
			src.writeBytes(pt);
			ofb.encrypt(src);
			assert("OFB_AES128 test 1", Hex.fromArray(src)==Hex.fromArray(ct));
			ofb.decrypt(src);
			assert("OFB_AES128 test 2", Hex.fromArray(src)==Hex.fromArray(pt));
		}
		
		public function testOFB_AES192():void {
			var key:ByteArray = Hex.toArray("8e73b0f7da0e6452c810f32b809079e562f8ead2522c6b7b");
			var pt:ByteArray = Hex.toArray(
				"6bc1bee22e409f96e93d7e117393172a" + 
				"ae2d8a571e03ac9c9eb76fac45af8e51" + 
				"30c81c46a35ce411e5fbc1191a0a52ef" + 
				"f69f2445df4f9b17ad2b417be66c3710");
			var ct:ByteArray = Hex.toArray(
				"cdc80d6fddf18cab34c25909c99a4174" + 
				"fcc28b8d4c63837c09e81700c1100401" + 
				"8d9a9aeac0f6596f559c6d4daf59a5f2" + 
				"6d9f200857ca6c3e9cac524bd9acc92a");
			var ofb:OFBMode = new OFBMode(new AESKey(key), new NullPad);
			ofb.IV = Hex.toArray("000102030405060708090a0b0c0d0e0f");
			var src:ByteArray = new ByteArray;
			src.writeBytes(pt);
			ofb.encrypt(src);
			assert("OFB_AES192 test 1", Hex.fromArray(src)==Hex.fromArray(ct));
			ofb.decrypt(src);
			assert("OFB_AES192 test 2", Hex.fromArray(src)==Hex.fromArray(pt));
		}
		
		public function testOFB_AES256():void {
			var key:ByteArray = Hex.toArray(
				"603deb1015ca71be2b73aef0857d7781" + 
				"1f352c073b6108d72d9810a30914dff4");
			var pt:ByteArray = Hex.toArray(
				"6bc1bee22e409f96e93d7e117393172a" + 
				"ae2d8a571e03ac9c9eb76fac45af8e51" + 
				"30c81c46a35ce411e5fbc1191a0a52ef" + 
				"f69f2445df4f9b17ad2b417be66c3710");
			var ct:ByteArray = Hex.toArray(
				"dc7e84bfda79164b7ecd8486985d3860" + 
				"4febdc6740d20b3ac88f6ad82a4fb08d" + 
				"71ab47a086e86eedf39d1c5bba97c408" + 
				"0126141d67f37be8538f5a8be740e484");
			var ofb:OFBMode = new OFBMode(new AESKey(key), new NullPad);
			ofb.IV = Hex.toArray("000102030405060708090a0b0c0d0e0f");
			var src:ByteArray = new ByteArray;
			src.writeBytes(pt);
			ofb.encrypt(src);
			assert("OFB_AES256 test 1", Hex.fromArray(src)==Hex.fromArray(ct));
			ofb.decrypt(src);
			assert("OFB_AES256 test 2", Hex.fromArray(src)==Hex.fromArray(pt));
		}
		
	}
}