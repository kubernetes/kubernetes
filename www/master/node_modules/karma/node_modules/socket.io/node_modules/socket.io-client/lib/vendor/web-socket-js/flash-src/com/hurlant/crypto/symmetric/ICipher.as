/**
 * ICipher
 * 
 * A generic interface to use symmetric ciphers
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.symmetric
{
	import flash.utils.ByteArray;
	
	public interface ICipher
	{
		function getBlockSize():uint;
		function encrypt(src:ByteArray):void;
		function decrypt(src:ByteArray):void;
		function dispose():void;
		function toString():String;
	}
}