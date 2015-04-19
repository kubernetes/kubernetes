/**
 * CFB8ModeTest
 * 
 * A test class for CFB8Mode
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.tests
{
	import flash.utils.ByteArray;
	import com.hurlant.util.Hex;
	import com.hurlant.crypto.symmetric.CFB8Mode;
	import com.hurlant.crypto.symmetric.AESKey;
	import com.hurlant.crypto.symmetric.NullPad;
	
	public class CFB8ModeTest extends TestCase
	{
		public function CFB8ModeTest(h:ITestHarness) {
			super(h,"CBF8Mode Test");
			runTest(testCFB8_AES128, "CFB-8 AES-128 Test Vectors");
			runTest(testCFB8_AES192, "CFB-8 AES-192 Test Vectors");
			runTest(testCFB8_AES256, "CFB-8 AES-192 Test Vectors");
			h.endTestCase();
		}
		
		/**
		 * http://csrc.nist.gov/publications/nistpubs/800-38a/sp800-38a.pdf
		 */
		public function testCFB8_AES128():void {
			var key:ByteArray = Hex.toArray("2b7e151628aed2a6abf7158809cf4f3c");
			var pt:ByteArray = Hex.toArray("6bc1bee22e409f96e93d7e117393172aae2d");
			var ct:ByteArray = Hex.toArray("3b79424c9c0dd436bace9e0ed4586a4f32b9");
			var cfb8:CFB8Mode = new CFB8Mode(new AESKey(key), new NullPad);
			cfb8.IV = Hex.toArray("000102030405060708090a0b0c0d0e0f");
			var src:ByteArray = new ByteArray;
			src.writeBytes(pt);
			cfb8.encrypt(src);
			assert("CFB8_AES128 test 1", Hex.fromArray(src)==Hex.fromArray(ct));
			cfb8.decrypt(src);
			assert("CFB8_AES128 test 2", Hex.fromArray(src)==Hex.fromArray(pt));
		}
		public function testCFB8_AES192():void {
			var key:ByteArray = Hex.toArray("8e73b0f7da0e6452c810f32b809079e562f8ead2522c6b7b");
			var pt:ByteArray = Hex.toArray("6bc1bee22e409f96e93d7e117393172aae2d");
			var ct:ByteArray = Hex.toArray("cda2521ef0a905ca44cd057cbf0d47a0678a");
			var cfb8:CFB8Mode = new CFB8Mode(new AESKey(key), new NullPad);
			cfb8.IV = Hex.toArray("000102030405060708090a0b0c0d0e0f");
			var src:ByteArray = new ByteArray;
			src.writeBytes(pt);
			cfb8.encrypt(src);
			assert("CFB8_AES128 test 1", Hex.fromArray(src)==Hex.fromArray(ct));
			cfb8.decrypt(src);
			assert("CFB8_AES128 test 2", Hex.fromArray(src)==Hex.fromArray(pt));
		}
		public function testCFB8_AES256():void {
			var key:ByteArray = Hex.toArray("603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4");
			var pt:ByteArray = Hex.toArray("6bc1bee22e409f96e93d7e117393172aae2d");
			var ct:ByteArray = Hex.toArray("dc1f1a8520a64db55fcc8ac554844e889700");
			var cfb8:CFB8Mode = new CFB8Mode(new AESKey(key), new NullPad);
			cfb8.IV = Hex.toArray("000102030405060708090a0b0c0d0e0f");
			var src:ByteArray = new ByteArray;
			src.writeBytes(pt);
			cfb8.encrypt(src);
			assert("CFB8_AES128 test 1", Hex.fromArray(src)==Hex.fromArray(ct));
			cfb8.decrypt(src);
			assert("CFB8_AES128 test 2", Hex.fromArray(src)==Hex.fromArray(pt));
		}
		
	}
}