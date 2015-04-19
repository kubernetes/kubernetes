/**
 * RSAKeyTest
 * 
 * A test class for RSAKey
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.tests
{
	import com.hurlant.crypto.rsa.RSAKey;
	import flash.utils.ByteArray;
	import com.hurlant.util.Hex;
	import com.hurlant.util.der.PEM;
	
	public class RSAKeyTest extends TestCase
	{
		public function RSAKeyTest(h:ITestHarness)
		{
			super(h, "RSA Testing");
			
			runTest(testSmoke,"RSA smoke test");
			runTest(testGenerate, "RSA Key Generation test");
			runTest(testPEM, "RSA Private Key PEM parsing");
			runTest(testPEM2, "RSA Public Key PEM parsing");
			h.endTestCase();
		}
		
		public function testSmoke():void {
			var N:String ="C4E3F7212602E1E396C0B6623CF11D26204ACE3E7D26685E037AD2507DCE82FC" + 
					"28F2D5F8A67FC3AFAB89A6D818D1F4C28CFA548418BD9F8E7426789A67E73E41";
			var E:String = "10001";
			var D:String = "7cd1745aec69096129b1f42da52ac9eae0afebbe0bc2ec89253598dcf454960e" + 
					"3e5e4ec9f8c87202b986601dd167253ee3fb3fa047e14f1dfd5ccd37e931b29d";
			var P:String = "f0e4dd1eac5622bd3932860fc749bbc48662edabdf3d2826059acc0251ac0d3b";
			var Q:String = "d13cb38fbcd06ee9bca330b4000b3dae5dae12b27e5173e4d888c325cda61ab3";
			var DMP1:String = "b3d5571197fc31b0eb6b4153b425e24c033b054d22b9c8282254fe69d8c8c593";
			var DMQ1:String = "968ffe89e50d7b72585a79b65cfdb9c1da0963cceb56c3759e57334de5a0ac3f";
			var IQMP:String = "d9bc4f420e93adad9f007d0e5744c2fe051c9ed9d3c9b65f439a18e13d6e3908";
			// create a key.
			var rsa:RSAKey = RSAKey.parsePrivateKey(N,E,D, P,Q,DMP1,DMQ1,IQMP);
			var txt:String = "hello";
			var src:ByteArray = Hex.toArray(Hex.fromString(txt));
			var dst:ByteArray = new ByteArray;
			var dst2:ByteArray = new ByteArray;
			rsa.encrypt(src, dst, src.length);
			rsa.decrypt(dst, dst2, dst.length);
			var txt2:String = Hex.toString(Hex.fromArray(dst2));
			assert("rsa encrypt+decrypt", txt==txt2);
		}
		
		public function testGenerate():void {
			var rsa:RSAKey = RSAKey.generate(256, "10001");
			// same lame smoke test here.
			var txt:String = "hello";
			var src:ByteArray = Hex.toArray(Hex.fromString(txt));
			var dst:ByteArray = new ByteArray;
			var dst2:ByteArray = new ByteArray;
			rsa.encrypt(src, dst, src.length);
			rsa.decrypt(dst, dst2, dst.length);
			var txt2:String = Hex.toString(Hex.fromArray(dst2));
			assert("rsa encrypt+decrypt", txt==txt2);
		}
		
		public function testPEM():void {
			var pem:String = "-----BEGIN RSA PRIVATE KEY-----\n" + 
					"MGQCAQACEQDJG3bkuB9Ie7jOldQTVdzPAgMBAAECEQCOGqcPhP8t8mX8cb4cQEaR\n" + 
					"AgkA5WTYuAGmH0cCCQDgbrto0i7qOQIINYr5btGrtccCCQCYy4qX4JDEMQIJAJll\n" + 
					"OnLVtCWk\n" + 
					"-----END RSA PRIVATE KEY-----";
			var rsa:RSAKey = PEM.readRSAPrivateKey(pem);
			//trace(rsa.dump());

			// obligatory use
			var txt:String = "hello";
			var src:ByteArray = Hex.toArray(Hex.fromString(txt));
			var dst:ByteArray = new ByteArray;
			var dst2:ByteArray = new ByteArray;
			rsa.encrypt(src, dst, src.length);
			rsa.decrypt(dst, dst2, dst.length);
			var txt2:String = Hex.toString(Hex.fromArray(dst2));
			assert("rsa encrypt+decrypt", txt==txt2);
		}
		public function testPEM2():void {
			var pem:String = "-----BEGIN PUBLIC KEY-----\n" + 
					"MCwwDQYJKoZIhvcNAQEBBQADGwAwGAIRAMkbduS4H0h7uM6V1BNV3M8CAwEAAQ==\n" + 
					"-----END PUBLIC KEY-----";
			var rsa:RSAKey = PEM.readRSAPublicKey(pem);
			//trace(rsa.dump());
		}
	}
}