package com.hurlant.math
{
	use namespace bi_internal;
	
	/**
	 * Modular reduction using "classic" algorithm
	 */
	internal class ClassicReduction implements IReduction
	{
		private var m:BigInteger;
		public function ClassicReduction(m:BigInteger) {
			this.m = m;
		}
		public function convert(x:BigInteger):BigInteger {
			if (x.s<0 || x.compareTo(m)>=0) {
				return x.mod(m);
			}
			return x;
		}
		public function revert(x:BigInteger):BigInteger {
			return x;
		}
		public function reduce(x:BigInteger):void {
			x.divRemTo(m, null,x);
		}
		public function mulTo(x:BigInteger, y:BigInteger, r:BigInteger):void {
			x.multiplyTo(y,r);
			reduce(r);
		}
		public function sqrTo(x:BigInteger, r:BigInteger):void {
			x.squareTo(r);
			reduce(r);
		}
	}
}