/**
 * PrintableString
 * 
 * An ASN1 type for a PrintableString, held within a String
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.util.der
{
	import flash.utils.ByteArray;
	
	public class PrintableString implements IAsn1Type
	{
		protected var type:uint;
		protected var len:uint;
		protected var str:String;
		
		public function PrintableString(type:uint, length:uint) {
			this.type = type;
			this.len = length;
		}
		
		public function getLength():uint
		{
			return len;
		}
		
		public function getType():uint
		{
			return type;
		}
		
		public function setString(s:String):void {
			str = s;
		}
		public function getString():String {
			return str;
		}
		
		public function toString():String {
			return DER.indent+str;
		}
		
		public function toDER():ByteArray {
			return null; // XXX not implemented
		}
	}
}