/**
 * Sequence
 * 
 * An ASN1 type for a Sequence, implemented as an Array
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.util.der
{
	import flash.utils.ByteArray;
	
	public dynamic class Sequence extends Array implements IAsn1Type
	{
		protected var type:uint;
		protected var len:uint;
		
		public function Sequence(type:uint = 0x30, length:uint = 0x00) {
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
			var tmp:ByteArray = new ByteArray;
			for (var i:int=0;i<length;i++) {
				var e:IAsn1Type = this[i];
				if (e == null) { // XXX Arguably, I could have a der.Null class instead
					tmp.writeByte(0x05);
					tmp.writeByte(0x00);
				} else {
					tmp.writeBytes(e.toDER());
				}
			}
			return DER.wrapDER(type, tmp);
		}
		
		public function toString():String {
			var s:String = DER.indent;
			DER.indent += "    ";
			var t:String = "";
			for (var i:int=0;i<length;i++) {
				if (this[i]==null) continue;
				var found:Boolean = false;
				for (var key:String in this) {
					if ( (i.toString()!=key) && this[i]==this[key]) {
						t += key+": "+this[i]+"\n";
						found = true;
						break;
					}
				}
				if (!found) t+=this[i]+"\n";
			}
//			var t:String = join("\n");
			DER.indent= s;
			return DER.indent+"Sequence["+type+"]["+len+"][\n"+t+"\n"+s+"]";
		}
		
		/////////
		
		public function findAttributeValue(oid:String):IAsn1Type {
			for each (var set:* in this) {
				if (set is Set) {
					var child:* = set[0];
					if (child is Sequence) {
						var tmp:* = child[0];
						if (tmp is ObjectIdentifier) {
							var id:ObjectIdentifier = tmp as ObjectIdentifier;
							if (id.toString()==oid) {
								return child[1] as IAsn1Type;
							}
						}
					}
				}
			}
			return null;
		}
		
		
	}
}