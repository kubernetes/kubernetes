var Buffer = require("buffer").Buffer;

function JSInflater(/*Buffer*/input) {

    var WSIZE = 0x8000,
        slide = new Buffer(0x10000),
        windowPos = 0,
        fixedTableList = null,
        fixedTableDist,
        fixedLookup,
        bitBuf = 0,
        bitLen = 0,
        method = -1,
        eof = false,
        copyLen = 0,
        copyDist = 0,
        tblList, tblDist, bitList, bitdist,

        inputPosition = 0,

        MASK_BITS = [0x0000, 0x0001, 0x0003, 0x0007, 0x000f, 0x001f, 0x003f, 0x007f, 0x00ff, 0x01ff, 0x03ff, 0x07ff, 0x0fff, 0x1fff, 0x3fff, 0x7fff, 0xffff],
        LENS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258, 0, 0],
        LEXT = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0, 99, 99],
        DISTS = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577],
        DEXT = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13],
        BITORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

    function HuffTable(clen, cnum, cval, blist, elist, lookupm) {

        this.status = 0;
        this.root = null;
        this.maxbit = 0;

        var el, f, tail,
            offsets = [],
            countTbl = [],
            sTbl = [],
            values = [],
            tentry = {extra: 0, bitcnt: 0, lbase: 0, next: null};

        tail = this.root = null;
        for(var i = 0; i < 0x11; i++)  { countTbl[i] = 0; sTbl[i] = 0; offsets[i] = 0; }
        for(i = 0; i < 0x120; i++) values[i] = 0;

        el = cnum > 256 ? clen[256] : 16;

        var pidx = -1;
        while (++pidx < cnum) countTbl[clen[pidx]]++;

        if(countTbl[0] == cnum) return;

        for(var j = 1; j <= 16; j++) if(countTbl[j] != 0) break;
        var bitLen = j;
        for(i = 16; i != 0; i--) if(countTbl[i] != 0) break;
        var maxLen = i;

        lookupm < j && (lookupm = j);

        var dCodes = 1 << j;
        for(; j < i; j++, dCodes <<= 1)
            if((dCodes -= countTbl[j]) < 0) {
                this.status = 2;
                this.maxbit = lookupm;
                return;
            }

        if((dCodes -= countTbl[i]) < 0) {
            this.status = 2;
            this.maxbit = lookupm;
            return;
        }

        countTbl[i] += dCodes;
        offsets[1] = j = 0;
        pidx = 1;
        var xp = 2;
        while(--i > 0) offsets[xp++] = (j += countTbl[pidx++]);
        pidx = 0;
        i = 0;
        do {
            (j = clen[pidx++]) && (values[offsets[j]++] = i);
        } while(++i < cnum);
        cnum = offsets[maxLen];
        offsets[0] = i = 0;
        pidx = 0;

        var level = -1,
            w = sTbl[0] = 0,
            cnode = null,
            tblCnt = 0,
            tblStack = [];

        for(; bitLen <= maxLen; bitLen++) {
            var kccnt = countTbl[bitLen];
            while(kccnt-- > 0) {
                while(bitLen > w + sTbl[1 + level]) {
                    w += sTbl[1 + level];
                    level++;
                    tblCnt = (tblCnt = maxLen - w) > lookupm ? lookupm : tblCnt;
                    if((f = 1 << (j = bitLen - w)) > kccnt + 1) {
                        f -= kccnt + 1;
                        xp = bitLen;
                        while(++j < tblCnt) {
                            if((f <<= 1) <= countTbl[++xp]) break;
                            f -= countTbl[xp];
                        }
                    }
                    if(w + j > el && w < el) j = el - w;
                    tblCnt = 1 << j;
                    sTbl[1 + level] = j;
                    cnode = [];
                    while (cnode.length < tblCnt) cnode.push({extra: 0, bitcnt: 0, lbase: 0, next: null});
                    if (tail == null) {
                        tail = this.root = {next:null, list:null};
                    } else {
                        tail = tail.next = {next:null, list:null}
                    }
                    tail.next = null;
                    tail.list = cnode;

                    tblStack[level] = cnode;

                    if(level > 0) {
                        offsets[level] = i;
                        tentry.bitcnt = sTbl[level];
                        tentry.extra = 16 + j;
                        tentry.next = cnode;
                        j = (i & ((1 << w) - 1)) >> (w - sTbl[level]);

                        tblStack[level-1][j].extra = tentry.extra;
                        tblStack[level-1][j].bitcnt = tentry.bitcnt;
                        tblStack[level-1][j].lbase = tentry.lbase;
                        tblStack[level-1][j].next = tentry.next;
                    }
                }
                tentry.bitcnt = bitLen - w;
                if(pidx >= cnum)
                    tentry.extra = 99;
                else if(values[pidx] < cval) {
                    tentry.extra = (values[pidx] < 256 ? 16 : 15);
                    tentry.lbase = values[pidx++];
                } else {
                    tentry.extra = elist[values[pidx] - cval];
                    tentry.lbase = blist[values[pidx++] - cval];
                }

                f = 1 << (bitLen - w);
                for(j = i >> w; j < tblCnt; j += f) {
                    cnode[j].extra = tentry.extra;
                    cnode[j].bitcnt = tentry.bitcnt;
                    cnode[j].lbase = tentry.lbase;
                    cnode[j].next = tentry.next;
                }
                for(j = 1 << (bitLen - 1); (i & j) != 0; j >>= 1)
                    i ^= j;
                i ^= j;
                while((i & ((1 << w) - 1)) != offsets[level]) {
                    w -= sTbl[level];
                    level--;
                }
            }
        }

        this.maxbit = sTbl[1];
        this.status = ((dCodes != 0 && maxLen != 1) ? 1 : 0);
    }

    function addBits(n) {
        while(bitLen < n) {
            bitBuf |= input[inputPosition++] << bitLen;
            bitLen += 8;
        }
        return bitBuf;
    }

    function cutBits(n) {
        bitLen -= n;
        return bitBuf >>= n;
    }

    function maskBits(n) {
        while(bitLen < n) {
            bitBuf |= input[inputPosition++] << bitLen;
            bitLen += 8;
        }
        var res = bitBuf & MASK_BITS[n];
        bitBuf >>= n;
        bitLen -= n;
        return res;
    }

    function codes(buff, off, size) {
        var e, t;
        if(size == 0) return 0;

        var n = 0;
        for(;;) {
            t = tblList.list[addBits(bitList) & MASK_BITS[bitList]];
            e = t.extra;
            while(e > 16) {
                if(e == 99) return -1;
                cutBits(t.bitcnt);
                e -= 16;
                t = t.next[addBits(e) & MASK_BITS[e]];
                e = t.extra;
            }
            cutBits(t.bitcnt);
            if(e == 16) {
                windowPos &= WSIZE - 1;
                buff[off + n++] = slide[windowPos++] = t.lbase;
                if(n == size) return size;
                continue;
            }
            if(e == 15) break;

            copyLen = t.lbase + maskBits(e);
            t = tblDist.list[addBits(bitdist) & MASK_BITS[bitdist]];
            e = t.extra;

            while(e > 16) {
                if(e == 99) return -1;
                cutBits(t.bitcnt);
                e -= 16;
                t = t.next[addBits(e) & MASK_BITS[e]];
                e = t.extra
            }
            cutBits(t.bitcnt);
            copyDist = windowPos - t.lbase - maskBits(e);

            while(copyLen > 0 && n < size) {
                copyLen--;
                copyDist &= WSIZE - 1;
                windowPos &= WSIZE - 1;
                buff[off + n++] = slide[windowPos++] = slide[copyDist++];
            }

            if(n == size) return size;
        }

        method = -1; // done
        return n;
    }

    function stored(buff, off, size) {
        cutBits(bitLen & 7);
        var n = maskBits(0x10);
        if(n != ((~maskBits(0x10)) & 0xffff)) return -1;
        copyLen = n;

        n = 0;
        while(copyLen > 0 && n < size) {
            copyLen--;
            windowPos &= WSIZE - 1;
            buff[off + n++] = slide[windowPos++] = maskBits(8);
        }

        if(copyLen == 0) method = -1;
        return n;
    }

    function fixed(buff, off, size) {
        var fixed_bd = 0;
        if(fixedTableList == null) {
            var lengths = [];

            for(var symbol = 0; symbol < 144; symbol++) lengths[symbol] = 8;
            for(; symbol < 256; symbol++) lengths[symbol] = 9;
            for(; symbol < 280; symbol++) lengths[symbol] = 7;
            for(; symbol < 288; symbol++) lengths[symbol] = 8;

            fixedLookup = 7;

            var htbl = new HuffTable(lengths, 288, 257, LENS, LEXT, fixedLookup);

            if(htbl.status != 0) return -1;

            fixedTableList = htbl.root;
            fixedLookup = htbl.maxbit;

            for(symbol = 0; symbol < 30; symbol++) lengths[symbol] = 5;
            fixed_bd = 5;

            htbl = new HuffTable(lengths, 30, 0, DISTS, DEXT, fixed_bd);
            if(htbl.status > 1) {
                fixedTableList = null;
                return -1;
            }
            fixedTableDist = htbl.root;
            fixed_bd = htbl.maxbit;
        }

        tblList = fixedTableList;
        tblDist = fixedTableDist;
        bitList = fixedLookup;
        bitdist = fixed_bd;
        return codes(buff, off, size);
    }

    function dynamic(buff, off, size) {
        var ll = new Array(0x023C);

        for (var m = 0; m < 0x023C; m++) ll[m] = 0;

        var llencnt = 257 + maskBits(5),
            dcodescnt = 1 + maskBits(5),
            bitlencnt = 4 + maskBits(4);

        if(llencnt > 286 || dcodescnt > 30) return -1;

        for(var j = 0; j < bitlencnt; j++) ll[BITORDER[j]] = maskBits(3);
        for(; j < 19; j++) ll[BITORDER[j]] = 0;

        // build decoding table for trees--single level, 7 bit lookup
        bitList = 7;
        var hufTable = new HuffTable(ll, 19, 19, null, null, bitList);
        if(hufTable.status != 0)
            return -1;	// incomplete code set

        tblList = hufTable.root;
        bitList = hufTable.maxbit;
        var lencnt = llencnt + dcodescnt,
            i = 0,
            lastLen = 0;
        while(i < lencnt) {
            var hufLcode = tblList.list[addBits(bitList) & MASK_BITS[bitList]];
            j = hufLcode.bitcnt;
            cutBits(j);
            j = hufLcode.lbase;
            if(j < 16)
                ll[i++] = lastLen = j;
            else if(j == 16) {
                j = 3 + maskBits(2);
                if(i + j > lencnt) return -1;
                while(j-- > 0) ll[i++] = lastLen;
            } else if(j == 17) {
                j = 3 + maskBits(3);
                if(i + j > lencnt) return -1;
                while(j-- > 0) ll[i++] = 0;
                lastLen = 0;
            } else {
                j = 11 + maskBits(7);
                if(i + j > lencnt) return -1;
                while(j-- > 0) ll[i++] = 0;
                lastLen = 0;
            }
        }
        bitList = 9;
        hufTable = new HuffTable(ll, llencnt, 257, LENS, LEXT, bitList);
        bitList == 0 && (hufTable.status = 1);

        if (hufTable.status != 0) return -1;

        tblList = hufTable.root;
        bitList = hufTable.maxbit;

        for(i = 0; i < dcodescnt; i++) ll[i] = ll[i + llencnt];
        bitdist = 6;
        hufTable = new HuffTable(ll, dcodescnt, 0, DISTS, DEXT, bitdist);
        tblDist = hufTable.root;
        bitdist = hufTable.maxbit;

        if((bitdist == 0 && llencnt > 257) || hufTable.status != 0) return -1;

        return codes(buff, off, size);
    }

    return {
        inflate : function(/*Buffer*/outputBuffer) {
            tblList = null;

            var size = outputBuffer.length,
                offset = 0, i;

            while(offset < size) {
                if(eof && method == -1) return;
                if(copyLen > 0) {
                    if(method != 0) {
                        while(copyLen > 0 && offset < size) {
                            copyLen--;
                            copyDist &= WSIZE - 1;
                            windowPos &= WSIZE - 1;
                            outputBuffer[offset++] = (slide[windowPos++] = slide[copyDist++]);
                        }
                    } else {
                        while(copyLen > 0 && offset < size) {
                            copyLen--;
                            windowPos &= WSIZE - 1;
                            outputBuffer[offset++] = (slide[windowPos++] = maskBits(8));
                        }
                        copyLen == 0 && (method = -1); // done
                    }
                    if (offset == size) return;
                }

                if(method == -1) {
                    if(eof) break;
                    eof = maskBits(1) != 0;
                    method = maskBits(2);
                    tblList = null;
                    copyLen = 0;
                }
                switch(method) {
                    case 0: i = stored(outputBuffer, offset, size - offset); break;
                    case 1: i = tblList != null ? codes(outputBuffer, offset, size - offset) : fixed(outputBuffer, offset, size - offset); break;
                    case 2: i = tblList != null ? codes(outputBuffer, offset, size - offset) : dynamic(outputBuffer, offset, size - offset); break;
                    default: i = -1; break;
                }

                if(i == -1) return;
                offset += i;
            }
        }
    };
}

module.exports = function(/*Buffer*/inbuf) {
    var zlib = require("zlib");
    return {
        inflateAsync : function(/*Function*/callback) {
            var tmp = zlib.createInflateRaw(),
                parts = [], total = 0;
            tmp.on('data', function(data) {
                parts.push(data);
                total += data.length;
            });
            tmp.on('end', function() {
                var buf = new Buffer(total), written = 0;
                buf.fill(0);

                for (var i = 0; i < parts.length; i++) {
                    var part = parts[i];
                    part.copy(buf, written);
                    written += part.length;
                }
                callback && callback(buf);
            });
            tmp.end(inbuf)
        },

        inflate : function(/*Buffer*/outputBuffer) {
            var x = {
                x: new JSInflater(inbuf)
            };
            x.x.inflate(outputBuffer);
            delete(x.x);
        }
    }
};
