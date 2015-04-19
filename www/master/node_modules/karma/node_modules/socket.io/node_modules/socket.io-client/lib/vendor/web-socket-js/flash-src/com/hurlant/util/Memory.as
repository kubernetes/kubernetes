/**
 * Memory
 * 
 * A class with a few memory-management methods, as much as 
 * such a thing exists in a Flash player.
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.util
{
	import flash.net.LocalConnection;
	import flash.system.System;
	
	public class Memory
	{
		public static function gc():void {
			// force a GC
			try {
			   new LocalConnection().connect('foo');
			   new LocalConnection().connect('foo');
			} catch (e:*) {}
		}
		public static function get used():uint {
			return System.totalMemory;
		}
	}
}