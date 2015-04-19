/**
 * ITestHarness
 * 
 * An interface to specify what's available for test cases to use.
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.tests
{
	public interface ITestHarness
	{
		function beginTestCase(name:String):void;
		function endTestCase():void;
		
		function beginTest(name:String):void;
		function passTest():void;
		function failTest(msg:String):void;
	}
}