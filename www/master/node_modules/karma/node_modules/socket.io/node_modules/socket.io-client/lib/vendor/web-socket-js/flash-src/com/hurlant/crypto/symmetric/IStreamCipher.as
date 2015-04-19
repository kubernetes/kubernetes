/**
 * IStreamCipher
 * 
 * A "marker" interface for stream ciphers.
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.symmetric {
	
	/**
	 * A marker to indicate how this cipher works.
	 * A stream cipher:
	 * - does not use initialization vector
	 * - keeps some internal state between calls to encrypt() and decrypt()
	 * 
	 */
	public interface IStreamCipher extends ICipher {
		
	}
}