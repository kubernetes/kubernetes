/* Base64 library for ActionScript 3.0.
 * Based on: Ma Bingyao code.
 * Optimized by: Jean-Philippe Auclair  / jpauclair.wordpress.com
 * Copyright (C) 2007 Ma Bingyao <andot@ujn.edu.cn>
 * LastModified: Oct 26, 2009
 * This library is free.  You can redistribute it and/or modify it.
 */
package com.hurlant.util{
    import flash.utils.ByteArray;

    public class Base64
    {
        private static const _encodeChars : Vector.<int> = InitEncoreChar();
        private static const _decodeChars : Vector.<int> = InitDecodeChar();

        public static function encodeByteArray(data : ByteArray) : String
        {
            var out : ByteArray = new ByteArray();
            //Presetting the length keep the memory smaller and optimize speed since there is no "grow" needed
            out.length = (2 + data.length - ((data.length + 2) % 3)) * 4 / 3; //Preset length //1.6 to 1.5 ms
            var i : int = 0;
            var r : int = data.length % 3;
            var len : int = data.length - r;
            var c : int;   //read (3) character AND write (4) characters

            while (i < len)
            {
                //Read 3 Characters (8bit * 3 = 24 bits)
                c = data[i++] << 16 | data[i++] << 8 | data[i++];

                //Cannot optimize this to read int because of the positioning overhead. (as3 bytearray seek is slow)
                //Convert to 4 Characters (6 bit * 4 = 24 bits)
                c = (_encodeChars[c >>> 18] << 24) | (_encodeChars[c >>> 12 & 0x3f] << 16) | (_encodeChars[c >>> 6 & 0x3f] << 8) | _encodeChars[c & 0x3f];

                //Optimization: On older and slower computer, do one write Int instead of 4 write byte: 1.5 to 0.71 ms
                out.writeInt(c);
                /*
                 out.writeByte(_encodeChars[c >> 18] );
                 out.writeByte(_encodeChars[c >> 12 & 0x3f]);
                 out.writeByte(_encodeChars[c >> 6 & 0x3f]);
                 out.writeByte(_encodeChars[c & 0x3f]);
                 */
            }

            if (r == 1) //Need two "=" padding
            {
                //Read one char, write two chars, write padding
                c = data[i];
                c = (_encodeChars[c >>> 2] << 24) | (_encodeChars[(c & 0x03) << 4] << 16) | 61 << 8 | 61;
                out.writeInt(c);
            }
            else if (r == 2) //Need one "=" padding
            {
                c = data[i++] << 8 | data[i];
                c = (_encodeChars[c >>> 10] << 24) | (_encodeChars[c >>> 4 & 0x3f] << 16) | (_encodeChars[(c & 0x0f) << 2] << 8) | 61;
                out.writeInt(c);
            }

            out.position = 0;
            return out.readUTFBytes(out.length);
        }


        public static function decodeToByteArray(str : String) : ByteArray
        {
            var c1 : int;
            var c2 : int;
            var c3 : int;
            var c4 : int;
            var i : int;
            var len : int;
            var out : ByteArray;
            len = str.length;
            i = 0;
            out = new ByteArray();
            var byteString : ByteArray = new ByteArray();
            byteString.writeUTFBytes(str);
            while (i < len)
            {
                //c1
                do
                {
                    c1 = _decodeChars[byteString[i++]];
                } while (i < len && c1 == -1);
                if (c1 == -1) break;

                //c2
                do
                {
                    c2 = _decodeChars[byteString[i++]];
                } while (i < len && c2 == -1);
                if (c2 == -1) break;

                out.writeByte((c1 << 2) | ((c2 & 0x30) >> 4));

                //c3
                do
                {
                    c3 = byteString[i++];
                    if (c3 == 61) return out;

                    c3 = _decodeChars[c3];
                } while (i < len && c3 == -1);
                if (c3 == -1) break;

                out.writeByte(((c2 & 0x0f) << 4) | ((c3 & 0x3c) >> 2));

                //c4
                do {
                    c4 = byteString[i++];
                    if (c4 == 61) return out;

                    c4 = _decodeChars[c4];
                } while (i < len && c4 == -1);
                if (c4 == -1) break;

                out.writeByte(((c3 & 0x03) << 6) | c4);

            }

            out.position = 0;

            return out;
        }

        public static function encode(data : String) : String {
            // Convert string to ByteArray
            var bytes : ByteArray = new ByteArray();
            bytes.writeUTFBytes(data);

            // Return encoded ByteArray
            return encodeByteArray(bytes);
        }

        public static function decode(data : String) : String {
            // Decode data to ByteArray
            var bytes : ByteArray = decodeToByteArray(data);

            // Convert to string and return
            return bytes.readUTFBytes(bytes.length);
        }

        public static function InitEncoreChar() : Vector.<int>
        {
            var encodeChars : Vector.<int> = new Vector.<int>();
            // We could push the number directly, but i think it's nice to see the characters (with no overhead on encode/decode)
            var chars : String = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            for (var i : int = 0; i < 64; i++)
            {
                encodeChars.push(chars.charCodeAt(i));
            }
            /*
             encodeChars.push(
             65, 66, 67, 68, 69, 70, 71, 72,
             73, 74, 75, 76, 77, 78, 79, 80,
             81, 82, 83, 84, 85, 86, 87, 88,
             89, 90, 97, 98, 99, 100, 101, 102,
             103, 104, 105, 106, 107, 108, 109, 110,
             111, 112, 113, 114, 115, 116, 117, 118,
             119, 120, 121, 122, 48, 49, 50, 51,
             52, 53, 54, 55, 56, 57, 43, 47);
             */
            return encodeChars;
        }

        public static function InitDecodeChar() : Vector.<int>
        {
            var decodeChars : Vector.<int> = new Vector.<int>();

            decodeChars.push(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 62, -1, -1, -1, 63,
                    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, -1, -1, -1, -1, -1, -1,
                    -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, -1, -1, -1, -1, -1,
                    -1, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, -1, -1, -1, -1, -1
                    - 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
            return decodeChars;
        }
    }
}