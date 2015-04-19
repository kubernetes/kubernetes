/**
 * NullPad
 * 
 * A padding class that doesn't pad.
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.symmetric
{
	import flash.utils.ByteArray;

	/**
	 * A pad that does nothing.
	 * Useful when you don't want padding in your Mode.
	 */
	public class NullPad implements IPad
	{
		public function unpad(a:ByteArray):void
		{
			return;
		}
		
		public function pad(a:ByteArray):void
		{
			return;
		}

		public function setBlockSize(bs:uint):void {
			return;
		}
		
	}
}