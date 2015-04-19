/**
 * DER
 * 
 * A basic class to parse DER structures.
 * It is very incomplete, but sufficient to extract whatever data we need so far.
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.util.der
{
	import com.hurlant.math.BigInteger;
	
	import flash.utils.ByteArray;
	import com.hurlant.util.der.Sequence;
	import com.hurlant.util.Hex;
	
	// goal 1: to be able to parse an RSA Private Key PEM file.
	// goal 2: to parse an X509v3 cert. kinda.
	
	/**
	 * DER for dummies:
	 * http://luca.ntop.org/Teaching/Appunti/asn1.html
	 * 
	 * This class does the bare minimum to get by. if that.
	 */
	public class DER
	{
		public static var indent:String = "";
		
		public static function parse(der:ByteArray, structure:*=null):IAsn1Type {
/* 			if (der.position==0) {
				trace("DER.parse: "+Hex.fromArray(der));
			}
 */			// type
			var type:int = der.readUnsignedByte();
			var constructed:Boolean = (type&0x20)!=0;
			type &=0x1F;
			// length
			var len:int = der.readUnsignedByte();
			if (len>=0x80) {
				// long form of length
				var count:int = len & 0x7f;
				len = 0;
				while (count>0) {
					len = (len<<8) | der.readUnsignedByte();
					count--;
				}
			}
			// data
			var b:ByteArray
			switch (type) {
				case 0x00: // WHAT IS THIS THINGY? (seen as 0xa0)
					// (note to self: read a spec someday.)
					// for now, treat as a sequence.
				case 0x10: // SEQUENCE/SEQUENCE OF. whatever
					// treat as an array
					var p:int = der.position;
					var o:Sequence = new Sequence(type, len);
					var arrayStruct:Array = structure as Array;
					if (arrayStruct!=null) {
						// copy the array, as we destroy it later.
						arrayStruct = arrayStruct.concat();
					}
					while (der.position < p+len) {
						var tmpStruct:Object = null
						if (arrayStruct!=null) {
							tmpStruct = arrayStruct.shift();
						}
						if (tmpStruct!=null) {
							while (tmpStruct && tmpStruct.optional) {
								// make sure we have something that looks reasonable. XXX I'm winging it here..
								var wantConstructed:Boolean = (tmpStruct.value is Array);
								var isConstructed:Boolean = isConstructedType(der);
								if (wantConstructed!=isConstructed) {
									// not found. put default stuff, or null
									o.push(tmpStruct.defaultValue);
									o[tmpStruct.name] = tmpStruct.defaultValue;
									// try the next thing
									tmpStruct = arrayStruct.shift();
								} else {
									break;
								}
							}
						}
						if (tmpStruct!=null) {
							var name:String = tmpStruct.name;
							var value:* = tmpStruct.value;
							if (tmpStruct.extract) {
								// we need to keep a binary copy of this element
								var size:int = getLengthOfNextElement(der);
								var ba:ByteArray = new ByteArray;
								ba.writeBytes(der, der.position, size);
								o[name+"_bin"] = ba;
							}
							var obj:IAsn1Type = DER.parse(der, value);
							o.push(obj);
							o[name] = obj;
						} else {
							o.push(DER.parse(der));
						}
					}
					return o;
				case 0x11: // SET/SET OF
					p = der.position;
					var s:Set = new Set(type, len);
					while (der.position < p+len) {
						s.push(DER.parse(der));
					}
					return s;
				case 0x02: // INTEGER
					// put in a BigInteger
					b = new ByteArray;
					der.readBytes(b,0,len);
					b.position=0;
					return new Integer(type, len, b);
				case 0x06: // OBJECT IDENTIFIER:
					b = new ByteArray;
					der.readBytes(b,0,len);
					b.position=0;
					return new ObjectIdentifier(type, len, b);
				default:
					trace("I DONT KNOW HOW TO HANDLE DER stuff of TYPE "+type);
					// fall through
				case 0x03: // BIT STRING
					if (der[der.position]==0) {
						//trace("Horrible Bit String pre-padding removal hack."); // I wish I had the patience to find a spec for this.
						der.position++;
						len--;
					}
				case 0x04: // OCTET STRING
					// stuff in a ByteArray for now.
					var bs:ByteString = new ByteString(type, len);
					der.readBytes(bs,0,len);
					return bs;
				case 0x05: // NULL
					// if len!=0, something's horribly wrong.
					// should I check?
					return null;
				case 0x13: // PrintableString
					var ps:PrintableString = new PrintableString(type, len);
					ps.setString(der.readMultiByte(len, "US-ASCII"));
					return ps;
				case 0x22: // XXX look up what this is. openssl uses this to store my email.
				case 0x14: // T61String - an horrible format we don't even pretend to support correctly
					ps = new PrintableString(type, len);
					ps.setString(der.readMultiByte(len, "latin1"));
					return ps;
				case 0x17: // UTCTime
					var ut:UTCTime = new UTCTime(type, len);
					ut.setUTCTime(der.readMultiByte(len, "US-ASCII"));
					return ut;
			}
		}
		
		private static function getLengthOfNextElement(b:ByteArray):int {
			var p:uint = b.position;
			// length
			b.position++;
			var len:int = b.readUnsignedByte();
			if (len>=0x80) {
				// long form of length
				var count:int = len & 0x7f;
				len = 0;
				while (count>0) {
					len = (len<<8) | b.readUnsignedByte();
					count--;
				}
			}
			len += b.position-p; // length of length
			b.position = p;
			return len;
		}
		private static function isConstructedType(b:ByteArray):Boolean {
			var type:int = b[b.position];
			return (type&0x20)!=0;
		}
		
		public static function wrapDER(type:int, data:ByteArray):ByteArray {
			var d:ByteArray = new ByteArray;
			d.writeByte(type);
			var len:int = data.length;
			if (len<128) {
				d.writeByte(len);
			} else if (len<256) {
				d.writeByte(1 | 0x80);
				d.writeByte(len);
			} else if (len<65536) {
				d.writeByte(2 | 0x80);
				d.writeByte(len>>8);
				d.writeByte(len);
			} else if (len<65536*256) {
				d.writeByte(3 | 0x80);
				d.writeByte(len>>16);
				d.writeByte(len>>8);
				d.writeByte(len);
			} else {
				d.writeByte(4 | 0x80);
				d.writeByte(len>>24);
				d.writeByte(len>>16);
				d.writeByte(len>>8);
				d.writeByte(len);
			}
			d.writeBytes(data);
			d.position=0;
			return d;
			
		}
	}
}