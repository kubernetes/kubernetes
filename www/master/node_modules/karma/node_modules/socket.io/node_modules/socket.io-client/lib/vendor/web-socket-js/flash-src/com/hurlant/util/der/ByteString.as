/**
 * ByteString
 * 
 * An ASN1 type for a ByteString, represented with a ByteArray
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.util.der
{
	import flash.utils.ByteArray;
	import com.hurlant.util.Hex;

	public class ByteString extends ByteArray implements IAsn1Type
	{
		private var type:uint;
		private var len:uint;
		
		public function ByteString(type:uint = 0x04, length:uint = 0x00) {
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
		
		public function toDER():ByteArray {
			return DER.wrapDER(type, this);
		}
		
		override public function toString():String {
			return DER.indent+"ByteString["+type+"]["+len+"]["+Hex.fromArray(this)+"]";
		}
		
	}
}