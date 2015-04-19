/**
 * BigIntegerTest
 * 
 * A test class for BigInteger
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.tests
{
	import com.hurlant.math.BigInteger;
	import com.hurlant.util.Hex;
	
	public class BigIntegerTest extends TestCase
	{
		public function BigIntegerTest(h:ITestHarness)
		{
			super(h, "BigInteger Tests");
			runTest(testAdd, "BigInteger Addition");
			h.endTestCase();
		}
		
		public function testAdd():void {
			var n1:BigInteger = BigInteger.nbv(25);
			var n2:BigInteger = BigInteger.nbv(1002);
			var n3:BigInteger = n1.add(n2);
			var v:int = n3.valueOf();
			assert("25+1002 = "+v, 25+1002==v);

			var p:BigInteger = new BigInteger(Hex.toArray("e564d8b801a61f47"));
			var xp:BigInteger = new BigInteger(Hex.toArray("99246db2a3507fa"));
			
			xp = xp.add(p);
			
			assert("xp==eef71f932bdb2741", xp.toString(16)=="eef71f932bdb2741");
		}
		
	}
}