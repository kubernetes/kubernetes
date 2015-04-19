/**
 * UTCTime
 * 
 * An ASN1 type for UTCTime, represented as a Date
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.util.der
{
	import flash.utils.ByteArray;
	
	public class UTCTime implements IAsn1Type
	{
		protected var type:uint;
		protected var len:uint;
		public var date:Date;
		
		public function UTCTime(type:uint, len:uint)
		{
			this.type = type;
			this.len = len;
		}
		
		public function getLength():uint
		{
			return len;
		}
		
		public function getType():uint
		{
			return type;
		}
		
		public function setUTCTime(str:String):void {
			
			var year:uint = parseInt(str.substr(0, 2));
			if (year<50) {
				year+=2000;
			} else {
				year+=1900;
			}
			var month:uint = parseInt(str.substr(2,2));
			var day:uint = parseInt(str.substr(4,2));
			var hour:uint = parseInt(str.substr(6,2));
			var minute:uint = parseInt(str.substr(8,2));
			// XXX this could be off by up to a day. parse the rest. someday.
			date = new Date(year, month-1, day, hour, minute);
		}
		
		
		public function toString():String {
			return DER.indent+"UTCTime["+type+"]["+len+"]["+date+"]";
		}
		
		public function toDER():ByteArray {
			return null // XXX not implemented
		}
	}
}