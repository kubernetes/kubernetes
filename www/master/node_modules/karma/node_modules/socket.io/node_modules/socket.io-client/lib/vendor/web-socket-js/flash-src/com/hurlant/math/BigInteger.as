/**
 * BigInteger
 * 
 * An ActionScript 3 implementation of BigInteger (light version)
 * Copyright (c) 2007 Henri Torgemane
 * 
 * Derived from:
 * 		The jsbn library, Copyright (c) 2003-2005 Tom Wu
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.math
{

	import com.hurlant.crypto.prng.Random;
	import com.hurlant.util.Hex;
	import com.hurlant.util.Memory;
	
	import flash.utils.ByteArray;
	use namespace bi_internal;

	public class BigInteger
	{
		public static const DB:int = 30; // number of significant bits per chunk
		public static const DV:int = (1<<DB);
		public static const DM:int = (DV-1); // Max value in a chunk
		
		public static const BI_FP:int = 52;
		public static const FV:Number = Math.pow(2, BI_FP);
		public static const F1:int = BI_FP - DB;
		public static const F2:int = 2*DB - BI_FP;
		
		public static const ZERO:BigInteger = nbv(0);
		public static const ONE:BigInteger  = nbv(1);
		
		/*bi_internal */public var t:int; // number of chunks.
		bi_internal var s:int; // sign
		bi_internal var a:Array; // chunks
		
		/**
		 * 
		 * @param value
		 * @param radix  WARNING: If value is ByteArray, this holds the number of bytes to use.
		 * @param unsigned
		 * 
		 */
		public function BigInteger(value:* = null, radix:int = 0, unsigned:Boolean = false) {
			a = new Array;
			if (value is String) {
				if (radix&&radix!=16) throw new Error("BigInteger construction with radix!=16 is not supported.");
				value = Hex.toArray(value);
				radix=0;
			}
			if (value is ByteArray) {
				var array:ByteArray = value as ByteArray;
				var length:int = radix || (array.length - array.position);
				fromArray(array, length, unsigned);
			}
		}
		public function dispose():void {
			var r:Random = new Random;
			for (var i:uint=0;i<a.length;i++) {
				a[i] = r.nextByte();
				delete a[i];
			}
			a=null;
			t=0;
			s=0;
			Memory.gc();
		}
		
		public function toString(radix:Number=16):String {
			if (s<0) return "-"+negate().toString(radix);
			var k:int;
			switch (radix) {
				case 2:   k=1; break;
				case 4:   k=2; break;
				case 8:   k=3; break;
				case 16:  k=4; break;
				case 32:  k=5; break;
				default:
//					return toRadix(radix);
			}
			var km:int = (1<<k)-1;
			var d:int = 0;
			var m:Boolean = false;
			var r:String = "";
			var i:int = t;
			var p:int = DB-(i*DB)%k;
			if (i-->0) {
				if (p<DB && (d=a[i]>>p)>0) {
					m = true;
					r = d.toString(36);
				}
				while (i >= 0) {
					if (p<k) {
						d = (a[i]&((1<<p)-1))<<(k-p);
						d|= a[--i]>>(p+=DB-k);
					} else {
						d = (a[i]>>(p-=k))&km;
						if (p<=0) {
							p += DB;
							--i;
						}
					}
					if (d>0) {
						m = true;
					}
					if (m) {
						r += d.toString(36);
					}
				}
			}
			return m?r:"0";
		}
		public function toArray(array:ByteArray):uint {
			const k:int = 8;
			const km:int = (1<<8)-1;
			var d:int = 0;
			var i:int = t;
			var p:int = DB-(i*DB)%k;
			var m:Boolean = false;
			var c:int = 0;
			if (i-->0) {
				if (p<DB && (d=a[i]>>p)>0) {
					m = true;
					array.writeByte(d);
					c++;
				}
				while (i >= 0) {
					if (p<k) {
						d = (a[i]&((1<<p)-1))<<(k-p);
						d|= a[--i]>>(p+=DB-k);
					} else {
						d = (a[i]>>(p-=k))&km;
						if (p<=0) {
							p += DB;
							--i;
						}
					}
					if (d>0) {
						m = true;
					}
					if (m) {
						array.writeByte(d);
						c++;
					}
				}
			}
			return c;
		}
		/**
		 * best-effort attempt to fit into a Number.
		 * precision can be lost if it just can't fit.
		 */
		public function valueOf():Number {
			if (s==-1) {
				return -negate().valueOf();
			}
			var coef:Number = 1;
			var value:Number = 0;
			for (var i:uint=0;i<t;i++) {
				value += a[i]*coef;
				coef *= DV;
			}
			return value;
		}
		/**
		 * -this
		 */
		public function negate():BigInteger {
			var r:BigInteger = nbi();
			ZERO.subTo(this, r);
			return r;
		}
		/**
		 * |this|
		 */
		public function abs():BigInteger {
			return (s<0)?negate():this;
		}
		/**
		 * return + if this > v, - if this < v, 0 if equal
		 */
		public function compareTo(v:BigInteger):int {
			var r:int = s - v.s;
			if (r!=0) {
				return r;
			}
			var i:int = t;
			r = i-v.t;
			if (r!=0) {
				return r;
			}
			while (--i >=0) {
				r=a[i]-v.a[i];
				if (r != 0) return r;
			}
			return 0;
		}
		/**
		 * returns bit length of the integer x
		 */
		bi_internal function nbits(x:int):int {
			var r:int = 1;
			var t:int;
			if ((t=x>>>16) != 0) { x = t; r += 16; }
			if ((t=x>>8) != 0) { x = t; r += 8; }
			if ((t=x>>4) != 0) { x = t; r += 4; }
			if ((t=x>>2) != 0) { x = t; r += 2; }
			if ((t=x>>1) != 0) { x = t; r += 1; }
			return r;
		}
		/**
		 * returns the number of bits in this
		 */
		public function bitLength():int {
			if (t<=0) return 0;
			return DB*(t-1)+nbits(a[t-1]^(s&DM));
		}
		/**
		 * 
		 * @param v
		 * @return this % v
		 * 
		 */
		public function mod(v:BigInteger):BigInteger {
			var r:BigInteger = nbi();
			abs().divRemTo(v,null,r);
			if (s<0 && r.compareTo(ZERO)>0) {
				v.subTo(r,r);
			}
			return r;
		}
		/**
		 * this^e % m, 0 <= e < 2^32
		 */
		public function modPowInt(e:int, m:BigInteger):BigInteger {
			var z:IReduction;
			if (e<256 || m.isEven()) {
				z = new ClassicReduction(m);
			} else {
				z = new MontgomeryReduction(m);
			}
			return exp(e, z);
		}

		/**
		 * copy this to r
		 */
		bi_internal function copyTo(r:BigInteger):void {
			for (var i:int = t-1; i>=0; --i) {
				r.a[i] = a[i];
			}
			r.t = t;
			r.s = s;
		}
		/**
		 * set from integer value "value", -DV <= value < DV
		 */
		bi_internal function fromInt(value:int):void {
			t = 1;
			s = (value<0)?-1:0;
			if (value>0) {
				a[0] = value;
			} else if (value<-1) {
				a[0] = value+DV;
			} else {
				t = 0;
			}
		}
		/**
		 * set from ByteArray and length,
		 * starting a current position
		 * If length goes beyond the array, pad with zeroes.
		 */
		bi_internal function fromArray(value:ByteArray, length:int, unsigned:Boolean = false):void {
			var p:int = value.position;
			var i:int = p+length;
			var sh:int = 0;
			const k:int = 8;
			t = 0;
			s = 0;
			while (--i >= p) {
				var x:int = i<value.length?value[i]:0;
				if (sh == 0) {
					a[t++] = x;
				} else if (sh+k > DB) {
					a[t-1] |= (x&((1<<(DB-sh))-1))<<sh;
					a[t++] = x>>(DB-sh);
				} else {
					a[t-1] |= x<<sh;
				}
				sh += k;
				if (sh >= DB) sh -= DB;
			}
			if (!unsigned && (value[0]&0x80)==0x80) {
				s = -1;
				if (sh > 0) {
					a[t-1] |= ((1<<(DB-sh))-1)<<sh;
				}
			}
			clamp();
			value.position = Math.min(p+length,value.length);
		}
		/**
		 * clamp off excess high words
		 */
		bi_internal function clamp():void {
			var c:int = s&DM;
			while (t>0 && a[t-1]==c) {
				--t;
			}
		}
		/**
		 * r = this << n*DB
		 */
		bi_internal function dlShiftTo(n:int, r:BigInteger):void {
			var i:int;
			for (i=t-1; i>=0; --i) {
				r.a[i+n] = a[i];
			}
			for (i=n-1; i>=0; --i) {
				r.a[i] = 0;
			}
			r.t = t+n;
			r.s = s;
		}
		/**
		 * r = this >> n*DB
		 */
		bi_internal function drShiftTo(n:int, r:BigInteger):void {
			var i:int;
			for (i=n; i<t; ++i) {
				r.a[i-n] = a[i];
			}
			r.t = Math.max(t-n,0);
			r.s = s;
		}
		/**
		 * r = this << n
		 */
		bi_internal function lShiftTo(n:int, r:BigInteger):void {
			var bs:int = n%DB;
			var cbs:int = DB-bs;
			var bm:int = (1<<cbs)-1;
			var ds:int = n/DB;
			var c:int = (s<<bs)&DM;
			var i:int;
			for (i=t-1; i>=0; --i) {
				r.a[i+ds+1] = (a[i]>>cbs)|c;
				c = (a[i]&bm)<<bs;
			}
			for (i=ds-1; i>=0; --i) {
				r.a[i] = 0;
			}
			r.a[ds] = c;
			r.t = t+ds+1;
			r.s = s;
			r.clamp();
		}
		/**
		 * r = this >> n
		 */
		bi_internal function rShiftTo(n:int, r:BigInteger):void {
			r.s = s;
			var ds:int = n/DB;
			if (ds >= t) {
				r.t = 0;
				return;
			}
			var bs:int = n%DB;
			var cbs:int = DB-bs;
			var bm:int = (1<<bs)-1;
			r.a[0] = a[ds]>>bs;
			var i:int;
			for (i=ds+1; i<t; ++i) {
				r.a[i-ds-1] |= (a[i]&bm)<<cbs;
				r.a[i-ds] = a[i]>>bs;
			}
			if (bs>0) {
				r.a[t-ds-1] |= (s&bm)<<cbs;
			}
			r.t = t-ds;
			r.clamp();
		}
		/**
		 * r = this - v
		 */
		bi_internal function subTo(v:BigInteger, r:BigInteger):void {
			var i:int = 0;
			var c:int = 0;
			var m:int = Math.min(v.t, t);
			while (i<m) {
				c += a[i] - v.a[i];
				r.a[i++] = c & DM;
				c >>= DB;
			}
			if (v.t < t) {
				c -= v.s;
				while (i< t) {
					c+= a[i];
					r.a[i++] = c&DM;
					c >>= DB;
				}
				c += s;
			} else {
				c += s;
				while (i < v.t) {
					c -= v.a[i];
					r.a[i++] = c&DM;
					c >>= DB;
				}
				c -= v.s;
			}
			r.s = (c<0)?-1:0;
			if (c<-1) {
				r.a[i++] = DV+c;
			} else if (c>0) {
				r.a[i++] = c;
			}
			r.t = i;
			r.clamp();
		}
		/**
		 * am: Compute w_j += (x*this_i), propagates carries,
		 * c is initial carry, returns final carry.
		 * c < 3*dvalue, x < 2*dvalue, this_i < dvalue
		 */
		bi_internal function am(i:int,x:int,w:BigInteger,j:int,c:int,n:int):int {
			var xl:int = x&0x7fff;
			var xh:int = x>>15;
			while(--n >= 0) {
				var l:int = a[i]&0x7fff;
				var h:int = a[i++]>>15;
				var m:int = xh*l + h*xl;
				l = xl*l + ((m&0x7fff)<<15)+w.a[j]+(c&0x3fffffff);
				c = (l>>>30)+(m>>>15)+xh*h+(c>>>30);
				w.a[j++] = l&0x3fffffff;
			}
			return c;
		}
		/**
		 * r = this * v, r != this,a (HAC 14.12)
		 * "this" should be the larger one if appropriate
		 */
		bi_internal function multiplyTo(v:BigInteger, r:BigInteger):void {
			var x:BigInteger = abs();
			var y:BigInteger = v.abs();
			var i:int = x.t;
			r.t = i+y.t;
			while (--i >= 0) {
				r.a[i] = 0;
			}
			for (i=0; i<y.t; ++i) {
				r.a[i+x.t] = x.am(0, y.a[i], r, i, 0, x.t);
			}
			r.s = 0;
			r.clamp();
			if (s!=v.s) {
				ZERO.subTo(r, r);
			}
		}
		/**
		 * r = this^2, r != this (HAC 14.16)
		 */
		bi_internal function squareTo(r:BigInteger):void {
			var x:BigInteger = abs();
			var i:int = r.t = 2*x.t;
			while (--i>=0) r.a[i] = 0;
			for (i=0; i<x.t-1; ++i) {
				var c:int = x.am(i, x.a[i], r, 2*i, 0, 1);
				if ((r.a[i+x.t] += x.am(i+1, 2*x.a[i], r, 2*i+1, c, x.t-i-1)) >= DV) {
					r.a[i+x.t] -= DV;
					r.a[i+x.t+1] = 1;
				}
			}
			if (r.t>0) {
				r.a[r.t-1] += x.am(i, x.a[i], r, 2*i, 0, 1);
			}
			r.s = 0;
			r.clamp();
		}
		/**
		 * divide this by m, quotient and remainder to q, r (HAC 14.20)
		 * r != q, this != m. q or r may be null.
		 */
		bi_internal function divRemTo(m:BigInteger, q:BigInteger = null, r:BigInteger = null):void {
			var pm:BigInteger = m.abs();
			if (pm.t <= 0) return;
			var pt:BigInteger = abs();
			if (pt.t < pm.t) {
				if (q!=null) q.fromInt(0);
				if (r!=null) copyTo(r);
				return;
			}
			if (r==null) r = nbi();
			var y:BigInteger = nbi();
			var ts:int = s;
			var ms:int = m.s;
			var nsh:int = DB-nbits(pm.a[pm.t-1]); // normalize modulus
			if (nsh>0) {
				pm.lShiftTo(nsh, y);
				pt.lShiftTo(nsh, r);
			} else {
				pm.copyTo(y);
				pt.copyTo(r);
			}
			var ys:int = y.t;
			var y0:int = y.a[ys-1];
			if (y0==0) return;
			var yt:Number = y0*(1<<F1)+((ys>1)?y.a[ys-2]>>F2:0);
			var d1:Number = FV/yt;
			var d2:Number = (1<<F1)/yt;
			var e:Number = 1<<F2;
			var i:int = r.t;
			var j:int = i-ys;
			var t:BigInteger = (q==null)?nbi():q;
			y.dlShiftTo(j,t);
			if (r.compareTo(t)>=0) {
				r.a[r.t++] = 1;
				r.subTo(t,r);
			}
			ONE.dlShiftTo(ys,t);
			t.subTo(y,y); // "negative" y so we can replace sub with am later.
			while(y.t<ys) y.(y.t++, 0);
			while(--j >= 0) {
				// Estimate quotient digit
				var qd:int = (r.a[--i]==y0)?DM:Number(r.a[i])*d1+(Number(r.a[i-1])+e)*d2;
				if ((r.a[i]+= y.am(0, qd, r, j, 0, ys))<qd) { // Try it out
					y.dlShiftTo(j, t);
					r.subTo(t,r);
					while (r.a[i]<--qd) {
						r.subTo(t,r);
					}
				}
			}
			if (q!=null) {
				r.drShiftTo(ys,q);
				if (ts!=ms) {
					ZERO.subTo(q,q);
				}
			}
			r.t = ys;
			r.clamp();
			if (nsh>0) {
				r.rShiftTo(nsh, r); // Denormalize remainder
			}
			if (ts<0) {
				ZERO.subTo(r,r);
			}
		}
		/**
		 * return "-1/this % 2^DB"; useful for Mont. reduction
		 * justification:
		 *         xy == 1 (mod n)
		 *         xy =  1+km
		 * 	 xy(2-xy) = (1+km)(1-km)
		 * x[y(2-xy)] =  1-k^2.m^2
		 * x[y(2-xy)] == 1 (mod m^2)
		 * if y is 1/x mod m, then y(2-xy) is 1/x mod m^2
		 * should reduce x and y(2-xy) by m^2 at each step to keep size bounded
		 * [XXX unit test the living shit out of this.]
		 */
		bi_internal function invDigit():int {
			if (t<1) return 0;
			var x:int = a[0];
			if ((x&1)==0) return 0;
			var y:int = x&3; 							// y == 1/x mod 2^2
			y = (y*(2-(x&0xf )*y))             &0xf;	// y == 1/x mod 2^4
			y = (y*(2-(x&0xff)*y))             &0xff;	// y == 1/x mod 2^8
			y = (y*(2-(((x&0xffff)*y)&0xffff)))&0xffff;	// y == 1/x mod 2^16
			// last step - calculate inverse mod DV directly;
			// assumes 16 < DB <= 32 and assumes ability to handle 48-bit ints
			// XXX 48 bit ints? Whaaaa? is there an implicit float conversion in here?
			y = (y*(2-x*y%DV))%DV;	// y == 1/x mod 2^dbits
			// we really want the negative inverse, and -DV < y < DV
			return (y>0)?DV-y:-y;
		}
		/**
		 * true iff this is even
		 */
		bi_internal function isEven():Boolean {
			return ((t>0)?(a[0]&1):s) == 0;
		}
		/**
		 * this^e, e < 2^32, doing sqr and mul with "r" (HAC 14.79)
		 */
		bi_internal function exp(e:int, z:IReduction):BigInteger {
			if (e > 0xffffffff || e < 1) return ONE;
			var r:BigInteger = nbi();
			var r2:BigInteger = nbi();
			var g:BigInteger = z.convert(this);
			var i:int = nbits(e)-1;
			g.copyTo(r);
			while(--i>=0) {
				z.sqrTo(r, r2);
				if ((e&(1<<i))>0) {
					z.mulTo(r2,g,r);
				} else {
					var t:BigInteger = r;
					r = r2;
					r2 = t;
				}
				
			}
			return z.revert(r);
		}
		bi_internal function intAt(str:String, index:int):int {
			return parseInt(str.charAt(index), 36);
		}


		protected function nbi():* {
			return new BigInteger;
		}
		/**
		 * return bigint initialized to value
		 */
		public static function nbv(value:int):BigInteger {
			var bn:BigInteger = new BigInteger;
			bn.fromInt(value);
			return bn;
		}


		// Functions above are sufficient for RSA encryption.
		// The stuff below is useful for decryption and key generation

		public static const lowprimes:Array = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313,317,331,337,347,349,353,359,367,373,379,383,389,397,401,409,419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,509];
		public static const lplim:int = (1<<26)/lowprimes[lowprimes.length-1];


		public function clone():BigInteger {
			var r:BigInteger = new BigInteger;
			this.copyTo(r);
			return r;
		}
		
		/**
		 * 
		 * @return value as integer
		 * 
		 */
		public function intValue():int {
			if (s<0) {
				if (t==1) {
					return a[0]-DV;
				} else if (t==0) {
					return -1;
				}
			} else if (t==1) {
				return a[0];
			} else if (t==0) {
				return 0;
			}
			// assumes 16 < DB < 32
			return  ((a[1]&((1<<(32-DB))-1))<<DB)|a[0];
		}
		
		/**
		 * 
		 * @return value as byte
		 * 
		 */
		public function byteValue():int {
			return (t==0)?s:(a[0]<<24)>>24;
		}
		
		/**
		 * 
		 * @return value as short (assumes DB>=16)
		 * 
		 */
		public function shortValue():int {
			return (t==0)?s:(a[0]<<16)>>16;
		}
		
		/**
		 * 
		 * @param r
		 * @return x s.t. r^x < DV
		 * 
		 */
		protected function chunkSize(r:Number):int {
			return Math.floor(Math.LN2*DB/Math.log(r));
		}
		
		/**
		 * 
		 * @return 0 if this ==0, 1 if this >0
		 * 
		 */
		public function sigNum():int {
			if (s<0) {
				return -1;
			} else if (t<=0 || (t==1 && a[0]<=0)) {
				return 0;
			} else{
				return 1;
			}
		}
		
		/**
		 * 
		 * @param b: radix to use
		 * @return a string representing the integer converted to the radix.
		 * 
		 */
		protected function toRadix(b:uint=10):String {
			if (sigNum()==0 || b<2 || b>32) return "0";
			var cs:int = chunkSize(b);
			var a:Number = Math.pow(b, cs);
			var d:BigInteger = nbv(a);
			var y:BigInteger = nbi();
			var z:BigInteger = nbi();
			var r:String = "";
			divRemTo(d, y, z);
			while (y.sigNum()>0) {
				r = (a+z.intValue()).toString(b).substr(1) + r;
				y.divRemTo(d,y,z);
			}
			return z.intValue().toString(b) + r;
		}
		
		/**
		 * 
		 * @param s a string to convert from using radix.
		 * @param b a radix
		 * 
		 */
		protected function fromRadix(s:String, b:int = 10):void {
			fromInt(0);
			var cs:int = chunkSize(b);
			var d:Number = Math.pow(b, cs);
			var mi:Boolean = false;
			var j:int = 0;
			var w:int = 0;
			for (var i:int=0;i<s.length;++i) {
				var x:int = intAt(s, i);
				if (x<0) {
					if (s.charAt(i) == "-" && sigNum() == 0) {
						mi = true;
					}
					continue;
				}
				w = b*w+x;
				if (++j >= cs) {
					dMultiply(d);
					dAddOffset(w,0);
					j=0;
					w=0;
				}
			}
			if (j>0) {
				dMultiply(Math.pow(b,j));
				dAddOffset(w,0);
			}
			if (mi) {
				BigInteger.ZERO.subTo(this, this);
			}
		}
		
		// XXX function fromNumber not written yet.
		
		/**
		 * 
		 * @return a byte array.
		 * 
		 */
		public function toByteArray():ByteArray {
			var i:int = t;
			var r:ByteArray = new ByteArray;
			r[0] = s;
			var p:int = DB-(i*DB)%8;
			var d:int;
			var k:int=0;
			if (i-->0) {
				if (p<DB && (d=a[i]>>p)!=(s&DM)>>p) {
					r[k++] = d|(s<<(DB-p));
				}
				while (i>=0) {
					if(p<8) {
						d = (a[i]&((1<<p)-1))<<(8-p);
						d|= a[--i]>>(p+=DB-8);
					} else {
						d = (a[i]>>(p-=8))&0xff;
						if (p<=0) {
							p += DB;
							--i;
						}
					}
					if ((d&0x80)!=0) d|=-256;
					if (k==0 && (s&0x80)!=(d&0x80)) ++k;
					if (k>0 || d!=s) r[k++] = d;
				} 
			}
			return r;
		}

		public function equals(a:BigInteger):Boolean {
			return compareTo(a)==0;
		}
		public function min(a:BigInteger):BigInteger {
			return (compareTo(a)<0)?this:a;
		}
		public function max(a:BigInteger):BigInteger {
			return (compareTo(a)>0)?this:a;
		}
		
		/**
		 * 
		 * @param a	a BigInteger to perform the operation with
		 * @param op a Function implementing the operation
		 * @param r a BigInteger to store the result of the operation
		 * 
		 */
		protected function bitwiseTo(a:BigInteger, op:Function, r:BigInteger):void {
			var i:int;
			var f:int;
			var m:int = Math.min(a.t, t);
			for (i=0; i<m; ++i) {
				r.a[i] = op(this.a[i],a.a[i]);
			}
			if (a.t<t) {
				f = a.s&DM;
				for (i=m;i<t;++i) {
					r.a[i] = op(this.a[i],f);
				}
				r.t = t;
			} else {
				f = s&DM;
				for (i=m;i<a.t;++i) {
					r.a[i] = op(f,a.a[i]);
				}
				r.t = a.t;
			}
			r.s = op(s, a.s);
			r.clamp();
		}
		
		private function op_and(x:int, y:int):int {return x&y;}
		public function and(a:BigInteger):BigInteger {
			var r:BigInteger = new BigInteger;
			bitwiseTo(a, op_and, r);
			return r;
		}
		
		private function op_or(x:int, y:int):int {return x|y;}
		public function or(a:BigInteger):BigInteger {
			var r:BigInteger = new BigInteger;
			bitwiseTo(a, op_or, r);
			return r;
		}
		
		private function op_xor(x:int, y:int):int {return x^y;}
		public function xor(a:BigInteger):BigInteger {
			var r:BigInteger = new BigInteger;
			bitwiseTo(a, op_xor, r);
			return r;
		}
		
		private function op_andnot(x:int, y:int):int { return x&~y;}
		public function andNot(a:BigInteger):BigInteger {
			var r:BigInteger = new BigInteger;
			bitwiseTo(a, op_andnot, r);
			return r;
		}
		
		public function not():BigInteger {
			var r:BigInteger = new BigInteger;
			for (var i:int=0;i<t;++i) {
				r[i] = DM&~a[i];
			}
			r.t = t;
			r.s = ~s;
			return r;
		}
		
		public function shiftLeft(n:int):BigInteger {
			var r:BigInteger = new BigInteger;
			if (n<0) {
				rShiftTo(-n, r);
			} else {
				lShiftTo(n, r);
			}
			return r;
		}
		public function shiftRight(n:int):BigInteger {
			var r:BigInteger = new BigInteger;
			if (n<0) {
				lShiftTo(-n, r);
			} else {
				rShiftTo(n, r);
			}
			return r;
		}
		
		/**
		 * 
		 * @param x
		 * @return index of lowet 1-bit in x, x < 2^31
		 * 
		 */
		private function lbit(x:int):int {
			if (x==0) return -1;
			var r:int = 0;
			if ((x&0xffff)==0) { x>>= 16; r += 16; }
			if ((x&0xff) == 0) { x>>=  8; r +=  8; }
			if ((x&0xf)  == 0) { x>>=  4; r +=  4; }
			if ((x&0x3)  == 0) { x>>=  2; r +=  2; }
			if ((x&0x1)  == 0) ++r;
			return r;
		}
		
		/**
		 * 
		 * @return index of lowest 1-bit (or -1 if none)
		 * 
		 */
		public function getLowestSetBit():int {
			for (var i:int=0;i<t;++i) {
				if (a[i]!=0) return i*DB+lbit(a[i]);
			}
			if (s<0) return t*DB;
			return -1;
		}
		
		/**
		 * 
		 * @param x
		 * @return number of 1 bits in x
		 * 
		 */
		private function cbit(x:int):int {
			var r:uint =0;
			while (x!=0) { x &= x-1; ++r }
			return r;
		}
		
		/**
		 * 
		 * @return number of set bits
		 * 
		 */
		public function bitCount():int {
			var r:int=0;
			var x:int = s&DM;
			for (var i:int=0;i<t;++i) {
				r += cbit(a[i]^x);
			}
			return r;
		}
		
		/**
		 * 
		 * @param n
		 * @return true iff nth bit is set
		 * 
		 */
		public function testBit(n:int):Boolean {
			var j:int = Math.floor(n/DB);
			if (j>=t) {
				return s!=0;
			}
			return ((a[j]&(1<<(n%DB)))!=0);
		}
		
		/**
		 * 
		 * @param n
		 * @param op
		 * @return this op (1<<n)
		 * 
		 */
		protected function changeBit(n:int,op:Function):BigInteger {
			var r:BigInteger = BigInteger.ONE.shiftLeft(n);
			bitwiseTo(r, op, r);
			return r;
		}
		
		/**
		 * 
		 * @param n
		 * @return this | (1<<n)
		 * 
		 */
		public function setBit(n:int):BigInteger { return changeBit(n, op_or); }

		/**
		 * 
		 * @param n
		 * @return this & ~(1<<n)
		 * 
		 */
		public function clearBit(n:int):BigInteger { return changeBit(n, op_andnot); }

		/**
		 * 
		 * @param n
		 * @return this ^ (1<<n)
		 * 
		 */
		public function flipBit(n:int):BigInteger { return changeBit(n, op_xor); }

		/**
		 * 
		 * @param a
		 * @param r = this + a
		 * 
		 */
		protected function addTo(a:BigInteger, r:BigInteger):void {
			var i:int = 0;
			var c:int = 0;
			var m:int = Math.min(a.t, t);
			while (i<m) {
				c += this.a[i] + a.a[i];
				r.a[i++] = c&DM;
				c>>=DB;
			}
			if (a.t < t) {
				c += a.s;
				while (i<t) {
					c += this.a[i];
					r.a[i++] = c&DM;
					c >>= DB;
				}
				c += s;
			} else {
				c += s;
				while (i<a.t) {
					c += a.a[i];
					r.a[i++] = c&DM;
					c >>= DB;
				}
				c += a.s;
			}
			r.s = (c<0)?-1:0;
			if (c>0) {
				r.a[i++] = c;
			} else if (c<-1) {
				r.a[i++] = DV+c;
			}
			r.t = i;
			r.clamp();
		}
		
		/**
		 * 
		 * @param a
		 * @return this + a
		 * 
		 */
		public function add(a:BigInteger):BigInteger {
			var r:BigInteger = new BigInteger;
			addTo(a,r);
			return r;
		}

		/**
		 * 
		 * @param a
		 * @return this - a
		 * 
		 */
		public function subtract(a:BigInteger):BigInteger {
			var r:BigInteger = new BigInteger;
			subTo(a,r);
			return r;
		}
		
		/**
		 * 
		 * @param a
		 * @return this * a
		 * 
		 */
		public function multiply(a:BigInteger):BigInteger {
			var r:BigInteger = new BigInteger;
			multiplyTo(a,r);
			return r;
		}
		
		/**
		 * 
		 * @param a
		 * @return this / a
		 * 
		 */
		public function divide(a:BigInteger):BigInteger {
			var r:BigInteger = new BigInteger;
			divRemTo(a, r, null);
			return r;
		}
		
		public function remainder(a:BigInteger):BigInteger {
			var r:BigInteger = new BigInteger;
			divRemTo(a, null, r);
			return r;
		}
		
		/**
		 * 
		 * @param a
		 * @return [this/a, this%a]
		 * 
		 */
		public function divideAndRemainder(a:BigInteger):Array {
			var q:BigInteger = new BigInteger;
			var r:BigInteger = new BigInteger;
			divRemTo(a, q, r);
			return [q,r];
		}
		
		/**
		 * 
		 * this *= n, this >=0, 1 < n < DV
		 * 
		 * @param n
		 * 
		 */
		bi_internal function dMultiply(n:int):void {
			a[t] = am(0, n-1, this, 0, 0, t);
			++t;
			clamp();
		}
		
		/**
		 * 
		 * this += n << w words, this >= 0
		 * 
		 * @param n
		 * @param w
		 * 
		 */
		bi_internal function dAddOffset(n:int, w:int):void {
			while (t<=w) {
				a[t++] = 0;
			}
			a[w] += n;
			while (a[w] >= DV) {
				a[w] -= DV;
				if (++w >= t) {
					a[t++] = 0;
				}
				++a[w];
			}
		}

		/**
		 * 
		 * @param e
		 * @return this^e
		 * 
		 */
		public function pow(e:int):BigInteger {
			return exp(e, new NullReduction);
		}
		
		/**
		 * 
		 * @param a
		 * @param n
		 * @param r = lower n words of "this * a", a.t <= n
		 * 
		 */
		bi_internal function multiplyLowerTo(a:BigInteger, n:int, r:BigInteger):void {
			var i:int = Math.min(t+a.t, n);
			r.s = 0; // assumes a, this >= 0
			r.t = i;
			while (i>0) {
				r.a[--i]=0;
			}
			var j:int;
			for (j=r.t-t;i<j;++i) {
				r.a[i+t] = am(0, a.a[i], r, i, 0, t);
			}
			for (j=Math.min(a.t,n);i<j;++i) {
				am(0, a.a[i], r, i, 0, n-i);
			}
			r.clamp();
		}
		
		/**
		 * 
		 * @param a
		 * @param n
		 * @param r = "this * a" without lower n words, n > 0
		 * 
		 */
		bi_internal function multiplyUpperTo(a:BigInteger, n:int, r:BigInteger):void {
			--n;
			var i:int = r.t = t+a.t-n;
			r.s = 0; // assumes a,this >= 0
			while (--i>=0) {
				r.a[i] = 0;
			}
			for (i=Math.max(n-t,0);i<a.t;++i) {
				r.a[t+i-n] = am(n-i, a.a[i], r, 0, 0, t+i-n);
			}
			r.clamp();
			r.drShiftTo(1,r);
		}
		
		/**
		 * 
		 * @param e
		 * @param m
		 * @return this^e % m (HAC 14.85)
		 * 
		 */
		public function modPow(e:BigInteger, m:BigInteger):BigInteger {
			var i:int = e.bitLength();
			var k:int;
			var r:BigInteger = nbv(1);
			var z:IReduction;
			
			if (i<=0) {
				return r;
			} else if (i<18) {
				k=1;
			} else if (i<48) {
				k=3;
			} else if (i<144) {
				k=4;
			} else if (i<768) {
				k=5;
			} else {
				k=6;
			}
			if (i<8) {
				z = new ClassicReduction(m);
			} else if (m.isEven()) {
				z = new BarrettReduction(m);
			} else {
				z = new MontgomeryReduction(m);
			}
			// precomputation
			var g:Array = [];
			var n:int = 3;
			var k1:int = k-1;
			var km:int = (1<<k)-1;
			g[1] = z.convert(this);
			if (k > 1) {
				var g2:BigInteger = new BigInteger;
				z.sqrTo(g[1], g2);
				while (n<=km) {
					g[n] = new BigInteger;
					z.mulTo(g2, g[n-2], g[n]);
					n += 2;
				}
			}
			
			var j:int = e.t-1;
			var w:int;
			var is1:Boolean = true;
			var r2:BigInteger = new BigInteger;
			var t:BigInteger;
			i = nbits(e.a[j])-1;
			while (j>=0) {
				if (i>=k1) {
					w = (e.a[j]>>(i-k1))&km;
				} else {
					w = (e.a[j]&((1<<(i+1))-1))<<(k1-i);
					if (j>0) {
						w |= e.a[j-1]>>(DB+i-k1);
					}
				}
				n = k;
				while ((w&1)==0) {
					w >>= 1;
					--n;
				}
				if ((i -= n) <0) {
					i += DB;
					--j;
				}
				if (is1) { // ret == 1, don't bother squaring or multiplying it
					g[w].copyTo(r);
					is1 = false;
				} else {
					while (n>1) {
						z.sqrTo(r, r2);
						z.sqrTo(r2, r);
						n -= 2;
					}
					if (n>0) {
						z.sqrTo(r, r2);
					} else {
						t = r;
						r = r2;
						r2 = t;
					}
					z.mulTo(r2, g[w], r);
				}
				while (j>=0 && (e.a[j]&(1<<i)) == 0) {
					z.sqrTo(r, r2);
					t = r;
					r = r2;
					r2 = t;
					if (--i<0) {
						i = DB-1;
						--j;
					}
					
				}
			}
			return z.revert(r);
		}
		
		/**
		 * 
		 * @param a
		 * @return gcd(this, a) (HAC 14.54)
		 * 
		 */
		public function gcd(a:BigInteger):BigInteger {
			var x:BigInteger = (s<0)?negate():clone();
			var y:BigInteger = (a.s<0)?a.negate():a.clone();
			if (x.compareTo(y)<0) {
				var t:BigInteger=x;
				x=y;
				y=t;
			}
			var i:int = x.getLowestSetBit();
			var g:int = y.getLowestSetBit();
			if (g<0) return x;
			if (i<g) g= i;
			if (g>0) {
				x.rShiftTo(g, x);
				y.rShiftTo(g, y);
			}
			while (x.sigNum()>0) {
				if ((i = x.getLowestSetBit()) >0) {
					x.rShiftTo(i, x);
				}
				if ((i = y.getLowestSetBit()) >0) {
					y.rShiftTo(i, y);
				}
				if (x.compareTo(y) >= 0) {
					x.subTo(y, x);
					x.rShiftTo(1, x);
				} else {
					y.subTo(x, y);
					y.rShiftTo(1, y);
				}
			}
			if (g>0) {
				y.lShiftTo(g, y);
			}
			return y;
		}

		/**
		 * 
		 * @param n
		 * @return this % n, n < 2^DB
		 * 
		 */
		protected function modInt(n:int):int {
			if (n<=0) return 0;
			var d:int = DV%n;
			var r:int = (s<0)?n-1:0;
			if (t>0) {
				if (d==0) {
					r = a[0]%n;
				} else {
					for (var i:int=t-1;i>=0;--i) {
						r = (d*r+a[i])%n;
					}
				}
			}
			return r;
		}
		
		/**
		 * 
		 * @param m
		 * @return 1/this %m (HAC 14.61)
		 * 
		 */
		public function modInverse(m:BigInteger):BigInteger {
			var ac:Boolean = m.isEven();
			if ((isEven()&&ac) || m.sigNum()==0) {
				return BigInteger.ZERO;
			}
			var u:BigInteger = m.clone();
			var v:BigInteger = clone();
			var a:BigInteger = nbv(1);
			var b:BigInteger = nbv(0);
			var c:BigInteger = nbv(0);
			var d:BigInteger = nbv(1);
			while (u.sigNum()!=0) {
				while (u.isEven()) {
					u.rShiftTo(1,u);
					if (ac) {
						if (!a.isEven() || !b.isEven()) {
							a.addTo(this,a);
							b.subTo(m,b);
						}
						a.rShiftTo(1,a);
					} else if (!b.isEven()) {
						b.subTo(m,b);
					}
					b.rShiftTo(1,b);
				}
				while (v.isEven()) {
					v.rShiftTo(1,v);
					if (ac) {
						if (!c.isEven() || !d.isEven()) {
							c.addTo(this,c);
							d.subTo(m,d);
						}
						c.rShiftTo(1,c);
					} else if (!d.isEven()) {
						d.subTo(m,d);
					}
					d.rShiftTo(1,d);
				}
				if (u.compareTo(v)>=0) {
					u.subTo(v,u);
					if (ac) {
						a.subTo(c,a);
					}
					b.subTo(d,b);
				} else {
					v.subTo(u,v);
					if (ac) {
						c.subTo(a,c);
					}
					d.subTo(b,d);
				}
			}
			if (v.compareTo(BigInteger.ONE) != 0) {
				return BigInteger.ZERO;
			}
			if (d.compareTo(m) >= 0) {
				return d.subtract(m);
			}
			if (d.sigNum()<0) {
				d.addTo(m,d);
			} else {
				return d;
			}
			if (d.sigNum()<0) {
				return d.add(m);
			} else {
				return d;
			}
		}

		/**
		 * 
		 * @param t
		 * @return primality with certainty >= 1-.5^t
		 * 
		 */
		public function isProbablePrime(t:int):Boolean {
			var i:int;
			var x:BigInteger = abs();
			if (x.t == 1 && x.a[0]<=lowprimes[lowprimes.length-1]) {
				for (i=0;i<lowprimes.length;++i) {
					if (x[0]==lowprimes[i]) return true;
				}
				return false;
			}
			if (x.isEven()) return false;
			i = 1;
			while (i<lowprimes.length) {
				var m:int = lowprimes[i];
				var j:int = i+1;
				while (j<lowprimes.length && m<lplim) {
					m *= lowprimes[j++];
				}
				m = x.modInt(m);
				while (i<j) {
					if (m%lowprimes[i++]==0) {
						return false;
					}
				}
			}
			return x.millerRabin(t);
		}
		
		/**
		 * 
		 * @param t
		 * @return true if probably prime (HAC 4.24, Miller-Rabin)
		 * 
		 */
		protected function millerRabin(t:int):Boolean {
			var n1:BigInteger = subtract(BigInteger.ONE);
			var k:int = n1.getLowestSetBit();
			if (k<=0) {
				return false;
			}
			var r:BigInteger = n1.shiftRight(k);
			t = (t+1)>>1;
			if (t>lowprimes.length) {
				t = lowprimes.length;
			}
			var a:BigInteger = new BigInteger;
			for (var i:int=0;i<t;++i) {
				a.fromInt(lowprimes[i]);
				var y:BigInteger = a.modPow(r, this);
				if (y.compareTo(BigInteger.ONE)!=0 && y.compareTo(n1)!=0) {
					var j:int = 1;
					while (j++<k && y.compareTo(n1)!=0) {
						y = y.modPowInt(2, this);
						if (y.compareTo(BigInteger.ONE)==0) {
							return false;
						}
					}
					if (y.compareTo(n1)!=0) {
						return false;
					}
				}
			}
			return true;
		}

		/**
		 * Tweak our BigInteger until it looks prime enough
		 * 
		 * @param bits
		 * @param t
		 * 
		 */
		public function primify(bits:int, t:int):void {
			if (!testBit(bits-1)) {	// force MSB set
				bitwiseTo(BigInteger.ONE.shiftLeft(bits-1), op_or, this);
			}
			if (isEven()) {
				dAddOffset(1,0);	// force odd
			}
			while (!isProbablePrime(t)) {
				dAddOffset(2,0);
				while(bitLength()>bits) subTo(BigInteger.ONE.shiftLeft(bits-1),this);
			}
		}

	}
}
