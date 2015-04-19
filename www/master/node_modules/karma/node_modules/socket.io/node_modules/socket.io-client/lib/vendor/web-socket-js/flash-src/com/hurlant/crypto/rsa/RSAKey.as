/**
 * RSAKey
 * 
 * An ActionScript 3 implementation of RSA + PKCS#1 (light version)
 * Copyright (c) 2007 Henri Torgemane
 * 
 * Derived from:
 * 		The jsbn library, Copyright (c) 2003-2005 Tom Wu
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.rsa
{
	import com.hurlant.crypto.prng.Random;
	import com.hurlant.math.BigInteger;
	import com.hurlant.util.Memory;
	
	import flash.utils.ByteArray;
	import com.hurlant.crypto.hash.IHash;
	import com.hurlant.util.Hex;
	import com.hurlant.util.der.DER;
	import com.hurlant.util.der.OID;
	import com.hurlant.util.ArrayUtil;
	import com.hurlant.util.der.Type;
	import com.hurlant.util.der.Sequence;
	import com.hurlant.util.der.ObjectIdentifier;
	import com.hurlant.util.der.ByteString;
	import com.hurlant.crypto.tls.TLSError;
	
	/**
	 * Current limitations:
	 * exponent must be smaller than 2^31.
	 */
	public class RSAKey
	{
		// public key
		public var e:int;              // public exponent. must be <2^31
		public var n:BigInteger; // modulus
		// private key
		public var d:BigInteger;
		// extended private key
		public var p:BigInteger;
		public var q:BigInteger;
		public var dmp1:BigInteger
		public var dmq1:BigInteger;
		public var coeff:BigInteger;
		// flags. flags are cool.
		protected var canDecrypt:Boolean;
		protected var canEncrypt:Boolean;
		
		public function RSAKey(N:BigInteger, E:int, 
			D:BigInteger=null,
			P:BigInteger = null, Q:BigInteger=null,
			DP:BigInteger=null, DQ:BigInteger=null,
			C:BigInteger=null) {
				
			this.n = N;
			this.e = E;
			this.d = D;
			this.p = P;
			this.q = Q;
			this.dmp1 = DP;
			this.dmq1 = DQ;
			this.coeff = C;
			
			// adjust a few flags.
			canEncrypt = (n!=null&&e!=0);
			canDecrypt = (canEncrypt&&d!=null);
			
			
		}

		public static function parsePublicKey(N:String, E:String):RSAKey {
			return new RSAKey(new BigInteger(N, 16, true), parseInt(E,16));
		}
		public static function parsePrivateKey(N:String, E:String, D:String, 
			P:String=null,Q:String=null, DMP1:String=null, DMQ1:String=null, IQMP:String=null):RSAKey {
			if (P==null) {
				return new RSAKey(new BigInteger(N,16, true), parseInt(E,16), new BigInteger(D,16, true));
			} else {
				return new RSAKey(new BigInteger(N,16, true), parseInt(E,16), new BigInteger(D,16, true),
					new BigInteger(P,16, true), new BigInteger(Q,16, true),
					new BigInteger(DMP1,16, true), new BigInteger(DMQ1, 16, true),
					new BigInteger(IQMP, 16, true));
			}
		}
		
		public function getBlockSize():uint {
			return (n.bitLength()+7)/8;
		}
		public function dispose():void {
			e = 0;
			n.dispose();
			n = null;
			Memory.gc();
		}

		public function encrypt(src:ByteArray, dst:ByteArray, length:uint, pad:Function=null):void {
			_encrypt(doPublic, src, dst, length, pad, 0x02);
		}
		public function decrypt(src:ByteArray, dst:ByteArray, length:uint, pad:Function=null):void {
			_decrypt(doPrivate2, src, dst, length, pad, 0x02);
		}

		public function sign(src:ByteArray, dst:ByteArray, length:uint, pad:Function = null):void {
			_encrypt(doPrivate2, src, dst, length, pad, 0x01);
		}
		public function verify(src:ByteArray, dst:ByteArray, length:uint, pad:Function = null):void {
			_decrypt(doPublic, src, dst, length, pad, 0x01);
		}
		

		private function _encrypt(op:Function, src:ByteArray, dst:ByteArray, length:uint, pad:Function, padType:int):void {
			// adjust pad if needed
			if (pad==null) pad = pkcs1pad;
			// convert src to BigInteger
			if (src.position >= src.length) {
				src.position = 0;
			}
			var bl:uint = getBlockSize();
			var end:int = src.position + length;
			while (src.position<end) {
				var block:BigInteger = new BigInteger(pad(src, end, bl, padType), bl, true);
				var chunk:BigInteger = op(block);
				chunk.toArray(dst);
			}
		}
		private function _decrypt(op:Function, src:ByteArray, dst:ByteArray, length:uint, pad:Function, padType:int):void {
			// adjust pad if needed
			if (pad==null) pad = pkcs1unpad;
			
			// convert src to BigInteger
			if (src.position >= src.length) {
				src.position = 0;
			}
			var bl:uint = getBlockSize();
			var end:int = src.position + length;
			while (src.position<end) {
				var block:BigInteger = new BigInteger(src, bl, true);
				var chunk:BigInteger = op(block);
				var b:ByteArray = pad(chunk, bl, padType);
				if (b == null) 
					 throw new TLSError( "Decrypt error - padding function returned null!", TLSError.decode_error );
				// if (b != null)
				dst.writeBytes(b);
			}
		}
		
		/**
		 * PKCS#1 pad. type 1 (0xff) or 2, random.
		 * puts as much data from src into it, leaves what doesn't fit alone.
		 */
		private function pkcs1pad(src:ByteArray, end:int, n:uint, type:uint = 0x02):ByteArray {
			var out:ByteArray = new ByteArray;
			var p:uint = src.position;
			end = Math.min(end, src.length, p+n-11);
			src.position = end;
			var i:int = end-1;
			while (i>=p && n>11) {
				out[--n] = src[i--];
			}
			out[--n] = 0;
			if (type==0x02) { // type 2
				var rng:Random = new Random;
				var x:int = 0;
				while (n>2) {
					do {
						x = rng.nextByte();
					} while (x==0);
					out[--n] = x;
				}
			} else { // type 1
				while (n>2) {
					out[--n] = 0xFF;
				}
			}
			out[--n] = type;
			out[--n] = 0;
			return out;
		}
		
		/**
		 * 
		 * @param src
		 * @param n
		 * @param type Not used.
		 * @return 
		 * 
		 */
		private function pkcs1unpad(src:BigInteger, n:uint, type:uint = 0x02):ByteArray {
			var b:ByteArray = src.toByteArray();
			var out:ByteArray = new ByteArray;
			
			b.position = 0;
			var i:int = 0;
			while (i<b.length && b[i]==0) ++i;
			if (b.length-i != n-1 || b[i]!=type) {
				trace("PKCS#1 unpad: i="+i+", expected b[i]=="+type+", got b[i]="+b[i].toString(16));
				return null;
			}
			++i;
			while (b[i]!=0) {
				if (++i>=b.length) {
					trace("PKCS#1 unpad: i="+i+", b[i-1]!=0 (="+b[i-1].toString(16)+")");
					return null;
				}
			}
			while (++i < b.length) {
				out.writeByte(b[i]);
			}
			out.position = 0;
			return out;
		}
		/**
		 * Raw pad.
		 */
		public function rawpad(src:ByteArray, end:int, n:uint, type:uint = 0):ByteArray {
			return src;
		}
		public function rawunpad(src:BigInteger, n:uint, type:uint = 0):ByteArray {
			return src.toByteArray();
		}
		
		public function toString():String {
			return "rsa";
		}
		
		public function dump():String {
			var s:String= "N="+n.toString(16)+"\n"+
			"E="+e.toString(16)+"\n";
			if (canDecrypt) {
				s+="D="+d.toString(16)+"\n";
				if (p!=null && q!=null) {
					s+="P="+p.toString(16)+"\n";
					s+="Q="+q.toString(16)+"\n";
					s+="DMP1="+dmp1.toString(16)+"\n";
					s+="DMQ1="+dmq1.toString(16)+"\n";
					s+="IQMP="+coeff.toString(16)+"\n";
				}
			}
			return s;
		}
		
		
		/**
		 * 
		 * note: We should have a "nice" variant of this function that takes a callback,
		 * 		and perform the computation is small fragments, to keep the web browser
		 * 		usable.
		 * 
		 * @param B
		 * @param E
		 * @return a new random private key B bits long, using public expt E
		 * 
		 */
		public static function generate(B:uint, E:String):RSAKey {
			var rng:Random = new Random;
			var qs:uint = B>>1;
			var key:RSAKey = new RSAKey(null,0,null);
			key.e = parseInt(E, 16);
			var ee:BigInteger = new BigInteger(E,16, true);
			for (;;) {
				for (;;) {
					key.p = bigRandom(B-qs, rng);
					if (key.p.subtract(BigInteger.ONE).gcd(ee).compareTo(BigInteger.ONE)==0 &&
						key.p.isProbablePrime(10)) break;
				}
				for (;;) {
					key.q = bigRandom(qs, rng);
					if (key.q.subtract(BigInteger.ONE).gcd(ee).compareTo(BigInteger.ONE)==0 &&
						key.q.isProbablePrime(10)) break;
				}
				if (key.p.compareTo(key.q)<=0) {
					var t:BigInteger = key.p;
					key.p = key.q;
					key.q = t;
				}
				var p1:BigInteger = key.p.subtract(BigInteger.ONE);
				var q1:BigInteger = key.q.subtract(BigInteger.ONE);
				var phi:BigInteger = p1.multiply(q1);
				if (phi.gcd(ee).compareTo(BigInteger.ONE)==0) {
					key.n = key.p.multiply(key.q);
					key.d = ee.modInverse(phi);
					key.dmp1 = key.d.mod(p1);
					key.dmq1 = key.d.mod(q1);
					key.coeff = key.q.modInverse(key.p);
					break;
				}
			}
			return key;
		}
		
		protected static function bigRandom(bits:int, rnd:Random):BigInteger {
			if (bits<2) return BigInteger.nbv(1);
			var x:ByteArray = new ByteArray;
			rnd.nextBytes(x, (bits>>3));
			x.position = 0;
			var b:BigInteger = new BigInteger(x,0,true);
			b.primify(bits, 1);
			return b;
		}
		
		protected function doPublic(x:BigInteger):BigInteger {
			return x.modPowInt(e, n);
		}
		
		protected function doPrivate2(x:BigInteger):BigInteger {
			if (p==null && q==null) {
				return x.modPow(d,n);
			}
			
			var xp:BigInteger = x.mod(p).modPow(dmp1, p);
			var xq:BigInteger = x.mod(q).modPow(dmq1, q);
			
			while (xp.compareTo(xq)<0) {
				xp = xp.add(p);
			}
			var r:BigInteger = xp.subtract(xq).multiply(coeff).mod(p).multiply(q).add(xq);
			
			return r;
		}
		
		protected function doPrivate(x:BigInteger):BigInteger {
			if (p==null || q==null) {
				return x.modPow(d, n);
			}
			// TODO: re-calculate any missing CRT params
			var xp:BigInteger = x.mod(p).modPow(dmp1, p);
			var xq:BigInteger = x.mod(q).modPow(dmq1, q);
			
			while (xp.compareTo(xq)<0) {
				xp = xp.add(p);
			}
			return xp.subtract(xq).multiply(coeff).mod(p).multiply(q).add(xq);
		}
		
		
	}
}