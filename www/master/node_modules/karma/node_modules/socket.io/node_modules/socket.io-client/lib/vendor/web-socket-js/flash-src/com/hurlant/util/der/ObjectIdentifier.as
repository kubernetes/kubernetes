/**
 * ObjectIdentifier
 * 
 * An ASN1 type for an ObjectIdentifier
 * We store the oid in an Array.
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.util.der
{
	import flash.utils.ByteArray;
	
	public class ObjectIdentifier implements IAsn1Type
	{
		private var type:uint;
		private var len:uint;
		private var oid:Array;
		
		public function ObjectIdentifier(type:uint, length:uint, b:*) {
			this.type = type;
			this.len = length;
			if (b is ByteArray) {
				parse(b as ByteArray);
			} else if (b is String) {
				generate(b as String);
			} else {
				throw new Error("Invalid call to new ObjectIdentifier");
			}
		}
		
		private function generate(s:String):void {
			oid = s.split(".");
		}
		
		private function parse(b:ByteArray):void {
			// parse stuff
			// first byte = 40*value1 + value2
			var o:uint = b.readUnsignedByte();
			var a:Array = []
			a.push(uint(o/40));
			a.push(uint(o%40));
			var v:uint = 0;
			while (b.bytesAvailable>0) {
				o = b.readUnsignedByte();
				var last:Boolean = (o&0x80)==0;
				o &= 0x7f;
				v = v*128 + o;
				if (last) {
					a.push(v);
					v = 0;
				}
			}
			oid = a;
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
			var tmp:Array = [];
			tmp[0] = oid[0]*40 + oid[1];
			for (var i:int=2;i<oid.length;i++) {
				var v:int = parseInt(oid[i]);
				if (v<128) {
					tmp.push(v);
				} else if (v<128*128) {
					tmp.push( (v>>7)|0x80 );
					tmp.push( v&0x7f );
				} else if (v<128*128*128) {
					tmp.push( (v>>14)|0x80 );
					tmp.push( (v>>7)&0x7f | 0x80 );
					tmp.push( v&0x7f);
				} else if (v<128*128*128*128) {
					tmp.push( (v>>21)|0x80 );
					tmp.push( (v>>14) & 0x7f | 0x80 );
					tmp.push( (v>>7) & 0x7f | 0x80 );
					tmp.push( v & 0x7f );
				} else {
					throw new Error("OID element bigger than we thought. :(");
				}
			}
			len = tmp.length;
			if (type==0) {
				type = 6;
			}
			tmp.unshift(len); // assume length is small enough to fit here.
			tmp.unshift(type);
			var b:ByteArray = new ByteArray;
			for (i=0;i<tmp.length;i++) {
				b[i] = tmp[i];
			}
			return b;
		}

		public function toString():String {
			return DER.indent+oid.join(".");
		}
		
		public function dump():String {
			return "OID["+type+"]["+len+"]["+toString()+"]";
		}
		
	}
}