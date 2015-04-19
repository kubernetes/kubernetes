/**
 * CTRModeTest
 * 
 * A test class for CTRMode
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.tests
{
	import flash.utils.ByteArray;
	import com.hurlant.util.Hex;
	import com.hurlant.crypto.symmetric.CTRMode;
	import com.hurlant.crypto.symmetric.AESKey;
	import com.hurlant.crypto.symmetric.NullPad;
	
	public class CTRModeTest extends TestCase
	{
		public function CTRModeTest(h:ITestHarness)
		{
			super(h, "CTRMode Test");
			runTest(testCTR_AES128, "CTR AES-128 Test Vectors");
			runTest(testCTR_AES192, "CTR AES-192 Test Vectors");
			runTest(testCTR_AES256, "CTR AES-256 Test Vectors");
			h.endTestCase();
		}
		
		/**
		 * Vectors from http://csrc.nist.gov/CryptoToolkit/modes/800-38_Series_Publications/SP800-38A.pdf
		 * Section F.5.1 and below.
		 */
		public function testCTR_AES128():void {
			var key:ByteArray = Hex.toArray("2b7e151628aed2a6abf7158809cf4f3c");
			var pt:ByteArray = Hex.toArray(
				"6bc1bee22e409f96e93d7e117393172a" +
				"ae2d8a571e03ac9c9eb76fac45af8e51" + 
				"30c81c46a35ce411e5fbc1191a0a52ef" + 
				"f69f2445df4f9b17ad2b417be66c3710");
			var ct:ByteArray = Hex.toArray(
				"874d6191b620e3261bef6864990db6ce" + 
				"9806f66b7970fdff8617187bb9fffdff" + 
				"5ae4df3edbd5d35e5b4f09020db03eab" + 
				"1e031dda2fbe03d1792170a0f3009cee");
			var ctr:CTRMode = new CTRMode(new AESKey(key), new NullPad);
			ctr.IV = Hex.toArray("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff");
			var src:ByteArray = new ByteArray;
			src.writeBytes(pt);
			ctr.encrypt(src);
			var hsrc:String = Hex.fromArray(src);
			var hct:String = Hex.fromArray(ct);
			assert("CTR_AES128 test 1: "+hsrc+" != "+hct, hsrc==hct);
			ctr.decrypt(src);
			hsrc = Hex.fromArray(src);
			var hpt:String = Hex.fromArray(pt);
			assert("CTR_AES128 test 2: "+hsrc+" != "+hpt, hsrc==hpt);
		}
		public function testCTR_AES192():void {
			var key:ByteArray = Hex.toArray("8e73b0f7da0e6452c810f32b809079e562f8ead2522c6b7b");
			var pt:ByteArray = Hex.toArray(
				"6bc1bee22e409f96e93d7e117393172a" +
				"ae2d8a571e03ac9c9eb76fac45af8e51" + 
				"30c81c46a35ce411e5fbc1191a0a52ef" + 
				"f69f2445df4f9b17ad2b417be66c3710");
			var ct:ByteArray = Hex.toArray(
				"1abc932417521ca24f2b0459fe7e6e0b" + 
				"090339ec0aa6faefd5ccc2c6f4ce8e94" + 
				"1e36b26bd1ebc670d1bd1d665620abf7" + 
				"4f78a7f6d29809585a97daec58c6b050");
			var ctr:CTRMode = new CTRMode(new AESKey(key), new NullPad);
			ctr.IV = Hex.toArray("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff");
			var src:ByteArray = new ByteArray;
			src.writeBytes(pt);
			ctr.encrypt(src);
			var hsrc:String = Hex.fromArray(src);
			var hct:String = Hex.fromArray(ct);
			assert("CTR_AES192 test 1: "+hsrc+" != "+hct, hsrc==hct);
			ctr.decrypt(src);
			hsrc = Hex.fromArray(src);
			var hpt:String = Hex.fromArray(pt);
			assert("CTR_AES192 test 2: "+hsrc+" != "+hpt, hsrc==hpt);
		}
		public function testCTR_AES256():void {
			var key:ByteArray = Hex.toArray("603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4");
			var pt:ByteArray = Hex.toArray(
				"6bc1bee22e409f96e93d7e117393172a" +
				"ae2d8a571e03ac9c9eb76fac45af8e51" + 
				"30c81c46a35ce411e5fbc1191a0a52ef" + 
				"f69f2445df4f9b17ad2b417be66c3710");
			var ct:ByteArray = Hex.toArray(
				"601ec313775789a5b7a7f504bbf3d228" + 
				"f443e3ca4d62b59aca84e990cacaf5c5" + 
				"2b0930daa23de94ce87017ba2d84988d" + 
				"dfc9c58db67aada613c2dd08457941a6");
			var ctr:CTRMode = new CTRMode(new AESKey(key), new NullPad);
			ctr.IV = Hex.toArray("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff");
			var src:ByteArray = new ByteArray;
			src.writeBytes(pt);
			ctr.encrypt(src);
			var hsrc:String = Hex.fromArray(src);
			var hct:String = Hex.fromArray(ct);
			assert("CTR_AES256 test 1: "+hsrc+" != "+hct, hsrc==hct);
			ctr.decrypt(src);
			hsrc = Hex.fromArray(src);
			var hpt:String = Hex.fromArray(pt);
			assert("CTR_AES256 test 2: "+hsrc+" != "+hpt, hsrc==hpt);
		}
		
	}
}