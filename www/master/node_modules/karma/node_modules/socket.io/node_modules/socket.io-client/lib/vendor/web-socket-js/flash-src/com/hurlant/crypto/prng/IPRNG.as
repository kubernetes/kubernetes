/**
 * IPRNG
 * 
 * An interface for classes that can be used a pseudo-random number generators
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.prng
{
	import flash.utils.ByteArray;
		
	public interface IPRNG {
		function getPoolSize():uint;
		function init(key:ByteArray):void;
		function next():uint;
		function dispose():void;
		function toString():String;
	}
}