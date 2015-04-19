/**
 * TestCase
 * 
 * Embryonic unit test support class.
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.tests
{
	public class TestCase 
	{
		public var harness:ITestHarness;
		
		public function TestCase(h:ITestHarness, title:String) {
			harness = h;
			harness.beginTestCase(title);
		}
		
		
		public function assert(msg:String, value:Boolean):void {
			if (value) {
//				TestHarness.print("+ ",msg);
				return;
			}
			throw new Error("Test Failure:"+msg);
		}
		
		public function runTest(f:Function, title:String):void {
			harness.beginTest(title);
			try {
				f();
			} catch (e:Error) {
				trace("EXCEPTION THROWN: "+e);
				trace(e.getStackTrace());
				harness.failTest(e.toString());
				return;
			}
			harness.passTest();
		}
	}
}