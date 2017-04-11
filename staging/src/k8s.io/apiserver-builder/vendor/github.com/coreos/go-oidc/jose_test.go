// +build !golint

// This file contains statically created JWKs for tests created by gen.go

package oidc

import (
	"encoding/json"

	jose "gopkg.in/square/go-jose.v2"
)

func mustLoadJWK(s string) jose.JSONWebKey {
	var jwk jose.JSONWebKey
	if err := json.Unmarshal([]byte(s), &jwk); err != nil {
		panic(err)
	}
	return jwk
}

var (
	testKeyECDSA_256_0 = mustLoadJWK(`{
		"kty": "EC",
		"kid": "bd06f3e11f523e310f8c2c8b20a892727fb558e0e23602312568b20a41e11188",
		"crv": "P-256",
		"x": "xK5N69f0-SAgWbjw2otcQeCGs3qqYMyqOWk4Os5Z_Xc",
		"y": "AXSaOPcMklJY9UKZhkGzVevqhAIUEzE3cfZ8o-ML5xE"
	}`)
	testKeyECDSA_256_0_Priv = mustLoadJWK(`{
		"kty": "EC",
		"kid": "bd06f3e11f523e310f8c2c8b20a892727fb558e0e23602312568b20a41e11188",
		"crv": "P-256",
		"x": "xK5N69f0-SAgWbjw2otcQeCGs3qqYMyqOWk4Os5Z_Xc",
		"y": "AXSaOPcMklJY9UKZhkGzVevqhAIUEzE3cfZ8o-ML5xE",
		"d": "L7jynYt-fMRPqw1e9vgXCGTg4yhGU4tlLxiFyNVimG4"
	}`)

	testKeyECDSA_256_1 = mustLoadJWK(`{
		"kty": "EC",
		"kid": "27ebd2f5bf6723e4de06caaab03703be458a37bf28a9fa748576a0826acb6ec2",
		"crv": "P-256",
		"x": "KZisP6wLCph4q6056jr7BH_asiX9RcLcS3HrNjdCpkw",
		"y": "5DrW-kEge0sePHlKmh1d2kqd10r32JEW6eyyewy18j8"
	}`)
	testKeyECDSA_256_1_Priv = mustLoadJWK(`{
		"kty": "EC",
		"kid": "27ebd2f5bf6723e4de06caaab03703be458a37bf28a9fa748576a0826acb6ec2",
		"crv": "P-256",
		"x": "KZisP6wLCph4q6056jr7BH_asiX9RcLcS3HrNjdCpkw",
		"y": "5DrW-kEge0sePHlKmh1d2kqd10r32JEW6eyyewy18j8",
		"d": "r6CiIpv0icIq5U4LYO39nBDVhCHCLObDFYC5IG9Y8Hk"
	}`)

	testKeyECDSA_256_2 = mustLoadJWK(`{
		"kty": "EC",
		"kid": "2c7d179db6006c90ece4d91f554791ff35156693c693102c00f66f473d15df17",
		"crv": "P-256",
		"x": "oDwcKp7SqgeRvycK5GgYjrlW4fbHn2Ybfd5iG7kDiPc",
		"y": "qazib9UwdUdbHSFzdy_HN10xZEItLvufPw0v7nIJOWA"
	}`)
	testKeyECDSA_256_2_Priv = mustLoadJWK(`{
		"kty": "EC",
		"kid": "2c7d179db6006c90ece4d91f554791ff35156693c693102c00f66f473d15df17",
		"crv": "P-256",
		"x": "oDwcKp7SqgeRvycK5GgYjrlW4fbHn2Ybfd5iG7kDiPc",
		"y": "qazib9UwdUdbHSFzdy_HN10xZEItLvufPw0v7nIJOWA",
		"d": "p79U6biKrOyrKzg-i3C7FVJiqzlqBhYQqmyOiZ9bhVM"
	}`)

	testKeyECDSA_256_3 = mustLoadJWK(`{
		"kty": "EC",
		"kid": "aba0a256999af470c8b2449103ae75b257d907da4d1cbfc73c1d45ab1098f544",
		"crv": "P-256",
		"x": "CKRYWVt1R7FzuJ43vEprfIzgB-KgIhRDhxLmd5ixiXY",
		"y": "QCxTVmK31ee710OYkNqdqEgHH3rqRQNQj3Wyq0xtYq0"
	}`)
	testKeyECDSA_256_3_Priv = mustLoadJWK(`{
		"kty": "EC",
		"kid": "aba0a256999af470c8b2449103ae75b257d907da4d1cbfc73c1d45ab1098f544",
		"crv": "P-256",
		"x": "CKRYWVt1R7FzuJ43vEprfIzgB-KgIhRDhxLmd5ixiXY",
		"y": "QCxTVmK31ee710OYkNqdqEgHH3rqRQNQj3Wyq0xtYq0",
		"d": "c_WflwwUxT_B5izk4iET49qoob0RH5hccnEScfBS9Qk"
	}`)

	testKeyECDSA_384_0 = mustLoadJWK(`{
		"kty": "EC",
		"kid": "bc271d0a68251171e0c5edfd384b32d154e1717af27c5089bda8fb598a474b7b",
		"crv": "P-384",
		"x": "FZEhxw06oB86xCAZCtKTfX3ze9tgxkdN199g-cWrpQTWF-2m6Blg1MN5D60Z9KoX",
		"y": "EyWjZ46gLlqfoU08iv0zcDWut1nbfoUoTd-7La2VY4PqQeDSFrxPCOyplxFMGJIW"
	}`)
	testKeyECDSA_384_0_Priv = mustLoadJWK(`{
		"kty": "EC",
		"kid": "bc271d0a68251171e0c5edfd384b32d154e1717af27c5089bda8fb598a474b7b",
		"crv": "P-384",
		"x": "FZEhxw06oB86xCAZCtKTfX3ze9tgxkdN199g-cWrpQTWF-2m6Blg1MN5D60Z9KoX",
		"y": "EyWjZ46gLlqfoU08iv0zcDWut1nbfoUoTd-7La2VY4PqQeDSFrxPCOyplxFMGJIW",
		"d": "t6oJHJBH2rvH3uyQkK_JwoUEwE1QHjwTnGLSDMZbNEDrEaR8BBiAo3s0p8rzGz5-"
	}`)

	testKeyECDSA_384_1 = mustLoadJWK(`{
		"kty": "EC",
		"kid": "58ef375d6e63cedc637fed59f4012a779cd749a83373237cf3972bba74ebd738",
		"crv": "P-384",
		"x": "Gj3zkiRR4RCJb0Tke3lD2spG2jzuXgX50fEwDZTRjlQIzz3Rc96Fw32FCSDectxQ",
		"y": "6alqN7ilqJAmqCU1BFrJJeivJiM0s1-RqBewRoNkTQinOLLbaiZlBAQxS4iq2dRv"
	}`)
	testKeyECDSA_384_1_Priv = mustLoadJWK(`{
		"kty": "EC",
		"kid": "58ef375d6e63cedc637fed59f4012a779cd749a83373237cf3972bba74ebd738",
		"crv": "P-384",
		"x": "Gj3zkiRR4RCJb0Tke3lD2spG2jzuXgX50fEwDZTRjlQIzz3Rc96Fw32FCSDectxQ",
		"y": "6alqN7ilqJAmqCU1BFrJJeivJiM0s1-RqBewRoNkTQinOLLbaiZlBAQxS4iq2dRv",
		"d": "-4eaoElyb_YANRownMtId_-glX2o45oc4_L_vgo3YOW5hxCq3KIBIdrhvBx1nw8B"
	}`)

	testKeyECDSA_384_2 = mustLoadJWK(`{
		"kty": "EC",
		"kid": "a8bab6f438b6cd2080711404549c978d1c00dd77a3c532bba12c964c4883b2ec",
		"crv": "P-384",
		"x": "CBKwYgZFLdTBBFnWD20q2YNUnRnOsDgTxG3y-dzCUrb65kOKm0ZFaZQPe5ZPjvDS",
		"y": "WlubxLv2qH-Aw-LsESKXAwm4HF3l4H1rVn3DZRpqcac6p-QSrXmfKCtxJVHDaTNi"
	}`)
	testKeyECDSA_384_2_Priv = mustLoadJWK(`{
		"kty": "EC",
		"kid": "a8bab6f438b6cd2080711404549c978d1c00dd77a3c532bba12c964c4883b2ec",
		"crv": "P-384",
		"x": "CBKwYgZFLdTBBFnWD20q2YNUnRnOsDgTxG3y-dzCUrb65kOKm0ZFaZQPe5ZPjvDS",
		"y": "WlubxLv2qH-Aw-LsESKXAwm4HF3l4H1rVn3DZRpqcac6p-QSrXmfKCtxJVHDaTNi",
		"d": "XfoOcxV3yfoAcRquJ9eaBvcY-H71B0XzXwx2eidolHDKLK7GHygEt9ToYSY_DvxO"
	}`)

	testKeyECDSA_384_3 = mustLoadJWK(`{
		"kty": "EC",
		"kid": "32f0cfc686455840530d727fb029002b9757ffff15272f98f53be9f77560718a",
		"crv": "P-384",
		"x": "xxED1z8EUCOUQ_jwS9nuVUDVzxs-U1rl19y8jMWrv4TdPeGHTRTgNUE57-YAL3ly",
		"y": "OsCN6-HmM-LP1itE5eW15WsSQoe3dZBX7AoHyKJUKtjzWKCJNUcyu3Np07xta1Cr"
	}`)
	testKeyECDSA_384_3_Priv = mustLoadJWK(`{
		"kty": "EC",
		"kid": "32f0cfc686455840530d727fb029002b9757ffff15272f98f53be9f77560718a",
		"crv": "P-384",
		"x": "xxED1z8EUCOUQ_jwS9nuVUDVzxs-U1rl19y8jMWrv4TdPeGHTRTgNUE57-YAL3ly",
		"y": "OsCN6-HmM-LP1itE5eW15WsSQoe3dZBX7AoHyKJUKtjzWKCJNUcyu3Np07xta1Cr",
		"d": "gfQA6wn4brWVT3OkeGiaCXGsQyTybZB9SdqTULsiSg8n6FS2T8hvK0doPwMTT5Gw"
	}`)

	testKeyECDSA_521_0 = mustLoadJWK(`{
		"kty": "EC",
		"kid": "f7b325c848a6c9fc5b72f131f179d2f37296837262797983c632597a4927726e",
		"crv": "P-521",
		"x": "AUaLqCAMEWPqiYgd-D_6F5kxpOgqnbQnjIHZ-NhjRnKKYuij9Iz7bq9pZU4F79wsODFpWxMFfISrveUfgEGt4Hy2",
		"y": "AeAKfmFsfcFttwQqv2B-fUfgLhw837YoGoNWh5qNE_LqTBxbYKRUkbSLxVRHEcVNnU1t3z9yMMdYXtuMlfJ0bhjr"
	}`)
	testKeyECDSA_521_0_Priv = mustLoadJWK(`{
		"kty": "EC",
		"kid": "f7b325c848a6c9fc5b72f131f179d2f37296837262797983c632597a4927726e",
		"crv": "P-521",
		"x": "AUaLqCAMEWPqiYgd-D_6F5kxpOgqnbQnjIHZ-NhjRnKKYuij9Iz7bq9pZU4F79wsODFpWxMFfISrveUfgEGt4Hy2",
		"y": "AeAKfmFsfcFttwQqv2B-fUfgLhw837YoGoNWh5qNE_LqTBxbYKRUkbSLxVRHEcVNnU1t3z9yMMdYXtuMlfJ0bhjr",
		"d": "AbD3guIvlVd8CZD0xUNuebgnQkE24XJnxCQ69P5VL3etEMmdr4HPJLPMQfMs10Gz8RTmrGKo-bdsU-cjS3d7dKIC"
	}`)

	testKeyECDSA_521_1 = mustLoadJWK(`{
		"kty": "EC",
		"kid": "4ea260ba2c9ed5fe9d6adeb998bb68b3edbab839d8bfd2be09fa628680793b90",
		"crv": "P-521",
		"x": "AE8-53o64GiywneanVZAHb96NcPbq0Ml6zynIcgLdKUiHGVs2te7SABq_9keFZJC6wooACeNWWT7VDK3kY77fjdh",
		"y": "AAnrIu0-D7JEd3-mqTR8Rdz53kw7DLAIypv0-_u4rqn0glwTZCkMpQ17wFH71bMInXaTi2Z_uq67NuVxFvUCaTvS"
	}`)
	testKeyECDSA_521_1_Priv = mustLoadJWK(`{
		"kty": "EC",
		"kid": "4ea260ba2c9ed5fe9d6adeb998bb68b3edbab839d8bfd2be09fa628680793b90",
		"crv": "P-521",
		"x": "AE8-53o64GiywneanVZAHb96NcPbq0Ml6zynIcgLdKUiHGVs2te7SABq_9keFZJC6wooACeNWWT7VDK3kY77fjdh",
		"y": "AAnrIu0-D7JEd3-mqTR8Rdz53kw7DLAIypv0-_u4rqn0glwTZCkMpQ17wFH71bMInXaTi2Z_uq67NuVxFvUCaTvS",
		"d": "AVVL9TIWcJCYH5r3QUhHXtsbkVXhLQSpp4sL1ta_H2_3bpRb9ZdVv10YOA-xN7Yz2wa-FMhIhj1ULe9z18ZM8dDF"
	}`)

	testKeyECDSA_521_2 = mustLoadJWK(`{
		"kty": "EC",
		"kid": "d69caddc821807ea63a46519b538a7d2f3135dbdda1ba628781f8567a47893d8",
		"crv": "P-521",
		"x": "AO618rH8GP78-mi9Z5FaPGqpyc_OVMK-BajZK-pwL89ZQdOvFa0fY0ENpB_KaRf5ELw9IP17lQh3T-O9O7jePDFj",
		"y": "AT1x9pCMbY-BXqAJPQDQxp6j8Gca7IpdxL8OS4td_XPiwXtX4JKcj_VKxtOw6k64yr_VuFYCs6wImOc72jNRBjA7"
	}`)
	testKeyECDSA_521_2_Priv = mustLoadJWK(`{
		"kty": "EC",
		"kid": "d69caddc821807ea63a46519b538a7d2f3135dbdda1ba628781f8567a47893d8",
		"crv": "P-521",
		"x": "AO618rH8GP78-mi9Z5FaPGqpyc_OVMK-BajZK-pwL89ZQdOvFa0fY0ENpB_KaRf5ELw9IP17lQh3T-O9O7jePDFj",
		"y": "AT1x9pCMbY-BXqAJPQDQxp6j8Gca7IpdxL8OS4td_XPiwXtX4JKcj_VKxtOw6k64yr_VuFYCs6wImOc72jNRBjA7",
		"d": "AYmtaW9ojb0Gb8zSqrlRnEKzRVIBIM5dsb1qWkd3mfr4Wl5tbPuiEctGLN9s6LDtY0JOL3nukOVoDbrmS4qCW64"
	}`)

	testKeyECDSA_521_3 = mustLoadJWK(`{
		"kty": "EC",
		"kid": "7193cba4fd9c72153133ce5cffc423857517ab8eb176a827e1a002eb5621edca",
		"crv": "P-521",
		"x": "AQfpLehZCER3ZTw1V1pN0RsX8-4WEW4IDxFDSGwrULQ79YHiLNmubrOSlSxiOSv2S-tHq-hxgma1PZlQRghJfemx",
		"y": "AAF41XT7jptLsy8FAaVRnex2WcSfdebcjjXMGO4rn6IlD3u9qnvrpR8MBp1gz5G1C5S7_NVgIeLSIYbdfd_wjVkd"
	}`)
	testKeyECDSA_521_3_Priv = mustLoadJWK(`{
		"kty": "EC",
		"kid": "7193cba4fd9c72153133ce5cffc423857517ab8eb176a827e1a002eb5621edca",
		"crv": "P-521",
		"x": "AQfpLehZCER3ZTw1V1pN0RsX8-4WEW4IDxFDSGwrULQ79YHiLNmubrOSlSxiOSv2S-tHq-hxgma1PZlQRghJfemx",
		"y": "AAF41XT7jptLsy8FAaVRnex2WcSfdebcjjXMGO4rn6IlD3u9qnvrpR8MBp1gz5G1C5S7_NVgIeLSIYbdfd_wjVkd",
		"d": "hzh0loqrhYFUik86SYBv3CC_GKNYankLMK95-Cfr1gLBD1l0M6W-7gn3XlyaQEInz0TBIbaQ1fL78HHjbVsNLrY"
	}`)

	testKeyRSA_1024_0 = mustLoadJWK(`{
		"kty": "RSA",
		"kid": "979d62e7114052027b11bcb51282d8228790134ef34e746f6372fa5aadaa9bf2",
		"n": "6zxmGE5X74SUBWfDEo8Tl-YxrkVfvxljQG9vKmNPQ2RhEmJ8eplZpKn9_nlxVDHGLzfJMqqUBS19EarIfDnOqyFBRkyKRbsQdUVU9XLjDIXqImJ8aN-UmShAYoeOClJVDsBJuxBGgS3pdgG7u3YQpvTV_hGuMoJr7UYgIrqb0qc",
		"e": "AQAB"
	}`)
	testKeyRSA_1024_0_Priv = mustLoadJWK(`{
		"kty": "RSA",
		"kid": "979d62e7114052027b11bcb51282d8228790134ef34e746f6372fa5aadaa9bf2",
		"n": "6zxmGE5X74SUBWfDEo8Tl-YxrkVfvxljQG9vKmNPQ2RhEmJ8eplZpKn9_nlxVDHGLzfJMqqUBS19EarIfDnOqyFBRkyKRbsQdUVU9XLjDIXqImJ8aN-UmShAYoeOClJVDsBJuxBGgS3pdgG7u3YQpvTV_hGuMoJr7UYgIrqb0qc",
		"e": "AQAB",
		"d": "IARggP5oyZjp7LJqwqPmrs4OBQI8Pe5eq-5-2u4ZY7rN24q8FpO4t8jLYU92NVdw-gxFvjepXesLEtSD5SSZFD7s-5rqmX9tW_t5t1a1sBY6IKz_YBSf2p2LtdTyrqgo4nRD0nYH3sMvGtOKCTV4K_vwfWPD3elXq752trG-HwE",
		"p": "9tgo12L3earNNrJfyIIDD2dqgKuNfWhQzhbe-Ju17dF37uGF6xnLqR0WXU-agGpUdYTMOd7IQi_bX_xkjQBYlw",
		"q": "8_YDvufPdccjGM2rAXs-2LbeP_LrzLUk1uGNpEjIt2lXEm6sQKjIfNLcAfVKh9zsqwqroveL2iI1ijsDgNAIcQ"
	}`)

	testKeyRSA_1024_1 = mustLoadJWK(`{
		"kty": "RSA",
		"kid": "4a3a37dc10aec5f427509e4014683102a0fafa32bec2ff9921ff19e1a55c79e8",
		"n": "sn7sU7dBEkU1RBIz_LKgrswSS7-68vTlOe7n-lanAqAlczm01_6IWrvcIC7lPv1iHQqWngusskANZirCZGTv6kK8kJLHBVRSROB9VkPTPNFYnSB6dacyPa0ty0otsaYOVM2RFvcCX7lBKKat2Tmst8vITdpvnoEzTgT_eGOKGAs",
		"e": "AQAB"
	}`)
	testKeyRSA_1024_1_Priv = mustLoadJWK(`{
		"kty": "RSA",
		"kid": "4a3a37dc10aec5f427509e4014683102a0fafa32bec2ff9921ff19e1a55c79e8",
		"n": "sn7sU7dBEkU1RBIz_LKgrswSS7-68vTlOe7n-lanAqAlczm01_6IWrvcIC7lPv1iHQqWngusskANZirCZGTv6kK8kJLHBVRSROB9VkPTPNFYnSB6dacyPa0ty0otsaYOVM2RFvcCX7lBKKat2Tmst8vITdpvnoEzTgT_eGOKGAs",
		"e": "AQAB",
		"d": "KTXqpE1sBaba7HNzc0Vemdzd4IVMyWlHPz_saTz2ZEHLQ7YwDapjmudCpF-PaCKiM2hNbAHwBluJfGwk4372cPHcJyri3Gs4GGj-cOlbXJh9aUp5o8fqn904B1UcPxMZ2DrZAKjseuLj3zyJgCsSJOGU1kJFRAkM8UN4a9cE8sE",
		"p": "3ntHMGDJEXwqldLNR-DucKGp32aJVAaKnn4j9qy1Tj6KhH70o3HyFVO_TKYxGg-pG3zTV-VzyMqmeh9hDdnyKw",
		"q": "zWMxEjyxvp-P-mNxhWQe1pJXZi6iPpcnt0SZLfweT3AtAzEaoSs4-4D1jT911LD4xDI8_a7-gYHgD8ME62PhoQ"
	}`)

	testKeyRSA_1024_2 = mustLoadJWK(`{
		"kty": "RSA",
		"kid": "d1c32aaea6c9527038e589dd94a546ca1beedb90f90e073228986cc258cf1fda",
		"n": "3mZrDaB5-7e29zok9XkGuu6FXYB00FqFgnGTJAGCMpql5uHz1h9p0DljZL4vsGkkYOZUMvqFS1pCEuzdSsupPNClf0NKMRux6yLv6iIR9C4pE9RBKPUrinzJuYs634rq5JOEP4IpP_fJfKxMw4Na85otd9KposKwP14cCkOibYM",
		"e": "AQAB"
	}`)
	testKeyRSA_1024_2_Priv = mustLoadJWK(`{
		"kty": "RSA",
		"kid": "d1c32aaea6c9527038e589dd94a546ca1beedb90f90e073228986cc258cf1fda",
		"n": "3mZrDaB5-7e29zok9XkGuu6FXYB00FqFgnGTJAGCMpql5uHz1h9p0DljZL4vsGkkYOZUMvqFS1pCEuzdSsupPNClf0NKMRux6yLv6iIR9C4pE9RBKPUrinzJuYs634rq5JOEP4IpP_fJfKxMw4Na85otd9KposKwP14cCkOibYM",
		"e": "AQAB",
		"d": "Ft2M0B_ZqsmepBh0SFCjIoD3cT-NwwYrh9fJewA0tKM1v2EnwrIEHQZpc6giGw8UUGod6gfbwH2NIYj8z33U7lyrKWPR7F8gJRm1KR5NLCBCnAnz0ukhsg24ktB25LZLZRCStRRhtCev95Vvmew0ip5081hv730Z2T_PsEyOU6E",
		"p": "8moFzLKJSQuhaIYTo1qNX0w8o4NBFb1atOlthHmq6Y6rdYTm_nyM9Q3mrNBolPS7LHTiBXtCEJ_m4V5hWDuGKw",
		"q": "6t0_mZKOoZ5NAE--MxkuOCcRhuqNo9VoacfPEH39CeoKAam4v5k56R7aQcb_raQK396pgV-evM2dpfdUaasiCQ"
	}`)

	testKeyRSA_1024_3 = mustLoadJWK(`{
		"kty": "RSA",
		"kid": "838742d0e3195b19f974fa30352db79070fa19aaaeb678ec95c499de7bd48d52",
		"n": "17Uc89-QvCjqBLXDJSCWxUoohjwFPI63Gub8g-lH1GSK3flXqiohz33KfKhqrdKsLrpRjskGTMg3Vo0IcLBnMYdp1i1nceORghtYLQsDNS7tqlHiKx725fLmWldGgiuP_0Ak0Knisw-j_q7Jx3OVAJnuS1o3vPLfJxJdyq6yV1k",
		"e": "AQAB"
	}`)
	testKeyRSA_1024_3_Priv = mustLoadJWK(`{
		"kty": "RSA",
		"kid": "838742d0e3195b19f974fa30352db79070fa19aaaeb678ec95c499de7bd48d52",
		"n": "17Uc89-QvCjqBLXDJSCWxUoohjwFPI63Gub8g-lH1GSK3flXqiohz33KfKhqrdKsLrpRjskGTMg3Vo0IcLBnMYdp1i1nceORghtYLQsDNS7tqlHiKx725fLmWldGgiuP_0Ak0Knisw-j_q7Jx3OVAJnuS1o3vPLfJxJdyq6yV1k",
		"e": "AQAB",
		"d": "B8cM-zIVauNixLa1CZKqPQTWfziMy8ktivfHJQ51O5BAfY5u_cC1JWEYuvPrnMba1Hh9VlOjOYOCk0lUg5OotMx5hxym6M5y_2rIrW90a8r3gGttPlZmyHQnFIgj2QZHlEyZGU1SIPTOtoECW5RGk7cYpA1s1_zfz8uyG-MiCPE",
		"p": "2dwia5iWWLZwFcCVylPh_7QDr3Wxt3kW_TmHIbaRT10Rborj1lAwltyEx7aVpHq-aNfvrYIEBbOXWwUU8PDUrQ",
		"q": "_XiDT55hn-Txi2C_peMVSDgXozf01qHKvdLirXM2uT_CI8kruYZYjWYn6_cSUptJ60eioM3WNTxjrSxWRS923Q"
	}`)

	testKeyRSA_2048_0 = mustLoadJWK(`{
		"kty": "RSA",
		"kid": "13b668c7b4f5d46d53de65e4b9f9c522fad4870e4d1656e459c88de6e90984d6",
		"n": "1j-CxBxkn20C_M24t6ueLp02T4MMAiPLXf5yaDcj7EcsXGbnsGEhrAUwCZwdEJAxeZ0PolmlE0XBOhQfRtsCVJmwu918aknptToyDbOUBr6WtPIK_c_BuGVanLCx3SnczV4jle9Bz7tGfpj2vAXytSBrfnZhdCHNEFeefQTQMnavfMhfWf_njiTa76BRyAHjb-XZIJHKovwBu0y3glmzhSKYNsUrW11RsWx6DbueWEbE3FpsHTiEdnJipcP3UKl3Z2z6t6n9ZYtFWkx4zCVVBQu-RWUQwjr2XnR1LFwXgL9xQocDBmS1O-wMTHqL3_oosNUdV3vuMPdxs_SEoys2rQ",
		"e": "AQAB"
	}`)
	testKeyRSA_2048_0_Priv = mustLoadJWK(`{
		"kty": "RSA",
		"kid": "13b668c7b4f5d46d53de65e4b9f9c522fad4870e4d1656e459c88de6e90984d6",
		"n": "1j-CxBxkn20C_M24t6ueLp02T4MMAiPLXf5yaDcj7EcsXGbnsGEhrAUwCZwdEJAxeZ0PolmlE0XBOhQfRtsCVJmwu918aknptToyDbOUBr6WtPIK_c_BuGVanLCx3SnczV4jle9Bz7tGfpj2vAXytSBrfnZhdCHNEFeefQTQMnavfMhfWf_njiTa76BRyAHjb-XZIJHKovwBu0y3glmzhSKYNsUrW11RsWx6DbueWEbE3FpsHTiEdnJipcP3UKl3Z2z6t6n9ZYtFWkx4zCVVBQu-RWUQwjr2XnR1LFwXgL9xQocDBmS1O-wMTHqL3_oosNUdV3vuMPdxs_SEoys2rQ",
		"e": "AQAB",
		"d": "C8F4X2Jfcw_8Nfrjw9A64bvmmv5JzmRAaGvpwyYjZneRS5Cp7demjVXLiPtz7NC8pjuj-_iHQkN1ksY_4RdrTVERjX1dskdT94m17WKJIMWcZ1lQmRSpQIDvM-HOIKCHaQ1dToDOT6Oq_o9OGosJAj9BJrNALasdIWRtYda9xcb7roLl_U3AtOlK9RiFygtt5uVBIPh1rsxaT0Y1MvMk4EMbFnv7NXXk65UM_3p2leSoPpO7LvlYs3WoRRX9ABH7zI-ppwLGEDuxfnwKzAOaRBe9oUh5rTYdazaY9XqofvekBc9Xqa2HpjkYX-L4Zy3oj04u-u5zX8KTh25jFC2a0Q",
		"p": "6L6icXqV7jLSXHrhMyLpW-tdFFp2XRQ_uKk972jmUJ-sEeh1YYSx_JpK6oaUnYshGbRLfEAOLy03iweAL4uMZJcASOR44TwSnn_qJZ72PkwhD42uKhE0eJRpxVht6jf5qhsynaMNUWc_FIW5ESpQIf6j4rqFV5WrKHAt0ypChTc",
		"q": "66fAza4rLvyjqQAIGQ07RiRYW4no1h4cK8RH_CDUgI3CCFQSuqTB8ZqM38r2u0tZoKzCNgHcnde6Jk07X6-LSQW2kypMXt3idbgaeVSYSbdZndgG_jTibJvoybxSjo6HfLpPvCrfsihXo7O5M13LE7VqBLbGTLI5iubOxcrfFTs"
	}`)

	testKeyRSA_2048_1 = mustLoadJWK(`{
		"kty": "RSA",
		"kid": "6d2b0efff0011bbbbf694de684b5b97cf454fd4086fa53b9af92545f3fa65459",
		"n": "1CmSBUbxU1jjnmaBG0r0cLLmyZgCOMbfpG6Z3HwfVDgCK6P-rU3F6QQazrOgGJJ0sz6vP50VK5u7BR6vSrPBBX5CicJPM2iNdz2JuV9ODEIkDQBLeI6TfIGhNOls-14tXKOPExY8b6JdyDMP31Hwo_0pF1FaunZ7yY1bgoKCtDV5-RKGd2EgylDGNPu0Ilr92MqCsAntBC8eQSkO4CcTcti4t9cX45VY5nPtwQRmp5zIgUHnU4LV3QVTLJnU-uidaAxRVQbS1pVql5xR6nYZHYvFk1IU-wTY6gGk5WvGWWQ440UTTaMAfnJP6VFDggUXeGSlKfKkDcz7JLT2Ma2KXQ",
		"e": "AQAB"
	}`)
	testKeyRSA_2048_1_Priv = mustLoadJWK(`{
		"kty": "RSA",
		"kid": "6d2b0efff0011bbbbf694de684b5b97cf454fd4086fa53b9af92545f3fa65459",
		"n": "1CmSBUbxU1jjnmaBG0r0cLLmyZgCOMbfpG6Z3HwfVDgCK6P-rU3F6QQazrOgGJJ0sz6vP50VK5u7BR6vSrPBBX5CicJPM2iNdz2JuV9ODEIkDQBLeI6TfIGhNOls-14tXKOPExY8b6JdyDMP31Hwo_0pF1FaunZ7yY1bgoKCtDV5-RKGd2EgylDGNPu0Ilr92MqCsAntBC8eQSkO4CcTcti4t9cX45VY5nPtwQRmp5zIgUHnU4LV3QVTLJnU-uidaAxRVQbS1pVql5xR6nYZHYvFk1IU-wTY6gGk5WvGWWQ440UTTaMAfnJP6VFDggUXeGSlKfKkDcz7JLT2Ma2KXQ",
		"e": "AQAB",
		"d": "DYJcHuPmh-UYEUT7oY5DRE3P7jQ0qALZyLGWMHji0c0DLl4x4D0chfrR7il33zisH6G1LPrGl1FCNlA-3yXU-5GPkRADVQWqRFZxx5Du-k7X1tAW_iUt9PaYGjNm0hasEsMDYDbBQGZ5TD8cGp8wEHEVRbvTaB4VQb8zfXrr8ad8XCFdtYQF5Sw_VuvUBcn-50kdl7S0MmQiwLx5xkisJHkVdVsrWbq-JJPMYrjYzPGo07kt8pQH2ecxrQD2waTzkLiAg-nytPBkAyMIFQ-XCulWCNiI-V_HJ0pqNctgVpdaSihlPCYhfllxUSU8yg4UvcfFEAN2S5yjvPyrJeJLAQ",
		"p": "4db4J-vxAyb_3otKidzYISIo8NuPxPXuYeImbvVbY6CWHdAYAkPgbwADh592D1vM6kwqZLot4VQ4XUVtX7Vf8tCaq5BAzcVAslK0rhK86kSmzuvtpS1ruzm1DGwJcQWOl_9qYnaAov8Ny0NG8fTm8iwvWJH7rVa3XNtPL0t_fx0",
		"q": "8H7_sX4kY3_4t90HbA8IvVkTf3-OGaIjtcDn2J3HLq1wMKAtdxwsbMtjhLYP70PPBe_FBNpcWhE7p-tOVhH1i0N0wwFqwlRQy-3Z4oiLeSsy0Npl29cfP7teO5UTrbrqfmHB0j2u_nEqA4n4iDF5QhPr_9yzxSR4Pg56z4PgFEE"
	}`)

	testKeyRSA_2048_2 = mustLoadJWK(`{
		"kty": "RSA",
		"kid": "7988e9dea9647281b04d094e4dc5737eaf739a412a570aa4d6deed7ee5640c30",
		"n": "wWOFgHHiJ54ZjQEKxvfqYDuhYymbfvYxvreroqx7E-cAuU4QBsOvKV3HNnH_vDRQE1AlqihNWmPFLptp7wxdMrLEtDnRFqTxyTXJ7wLCaaaB6Wwx1dhpr9QEG5-8rxLGFYv2w5i0-o76JPHG0tuqf0YNmHp9oWcv524XnDBuji4-u6km1KynT1EQKHYgy57JWgUpukgJGImBEvf0hC2Y5s5mHvVby3xAD9-cFa2o9Vj3G-SBYFsrRIVR8GcVNwo88oj8F7-2nGD_wOOu_qT_5I3vfToUYGj5-2rAK1ja0bZpXFibrkPrW4w9paG-3F7I0sPeL3e_BDC_rQ8TgL03uQ",
		"e": "AQAB"
	}`)
	testKeyRSA_2048_2_Priv = mustLoadJWK(`{
		"kty": "RSA",
		"kid": "7988e9dea9647281b04d094e4dc5737eaf739a412a570aa4d6deed7ee5640c30",
		"n": "wWOFgHHiJ54ZjQEKxvfqYDuhYymbfvYxvreroqx7E-cAuU4QBsOvKV3HNnH_vDRQE1AlqihNWmPFLptp7wxdMrLEtDnRFqTxyTXJ7wLCaaaB6Wwx1dhpr9QEG5-8rxLGFYv2w5i0-o76JPHG0tuqf0YNmHp9oWcv524XnDBuji4-u6km1KynT1EQKHYgy57JWgUpukgJGImBEvf0hC2Y5s5mHvVby3xAD9-cFa2o9Vj3G-SBYFsrRIVR8GcVNwo88oj8F7-2nGD_wOOu_qT_5I3vfToUYGj5-2rAK1ja0bZpXFibrkPrW4w9paG-3F7I0sPeL3e_BDC_rQ8TgL03uQ",
		"e": "AQAB",
		"d": "RPC4j-CJUbw_uY-Miv-oMuQvFU2o3Crh8u5BJn28ZozsKiMU_YRW9jUzJkqfczVm8muY8b7qTHXSvlmy-v_6XW9zRhhyXFMyypr9QNJIAifUmiTy4xwCGSdIy5w3RGY57UZ3EqVmpwe_TtpOGa8rabHMePX5wUcqwaLykcCGOPLOFKfPF93GqZYi2StS1IaiijLi2IAMhxQGqS6Ct3dA7yacQwegPiDzjb4Is3V9UQ9k_aS3I1w56tz5qspVmfDEuuWToc-2Qyk-eMKWl1tTDJuGTCiS5IFtMPerRpPq9GCpycV0csIW-6AnW79b038DxWjDAUsNnDPnX0y1dQ1G3Q",
		"p": "yh2KH_JqU29L2SLG5-ul1EBbaSbt3F5UUZbCvm_MIsiH1TzafVyBd8FeG1d-pMR8bgGjaDSA-DNkOEDAeAlLx-UCTl2Uk2ySoMnZr5JjVLW2DTuF7e8ySKrKQSSLe63aQyKuIZh9P1RNrwrmFwvJgXdzudVjirmdTwxAN0lijE8",
		"q": "9PJitAE_kk6SoKvJisXbDb7Xursj26xiYIdWuzlzQ1CbwYcJ1oORDpo0bEpy0BfIbZDiJJ79Xrh5Lo1BVysQ-IZJCXK7mavqiSa8g3B8rmqLXmAtjDbwLGsJOayCpYnsGgOLT7XIUYXCFH7C-tSSydhJ1j8JAbsJveMiBPKnUXc"
	}`)

	testKeyRSA_2048_3 = mustLoadJWK(`{
		"kty": "RSA",
		"kid": "d214aea9b78d15dd2667f5d19c36df6647fa50023140fd4f5866d285718b897c",
		"n": "qFnqWpBOZ_5jnIr_XkXnonmHII5gKzxPhUBfvWBhGN2eH8nmnGh6aUnKyKuCLP_qfYLU7cf9bah-e00451iGite8Tg9ZMYAPFX4NM5j0rGWNv9Z6lSn1xXezJv-FU9VUwXm0DG5eVcB8OV99JWxHivQUSBzY0Q3DUlgrD7FhMd7BrrcmTUO07KnW6SxN3oTbT7fNMxfJSTRxmvU-t-6sLhYP4Wbg39zEUgI8M7b7tvHp34klSqOn2DibVBhvWGF1-IgNT7ng6pS5iAZN7Cz7NjPc9ZM_IWyngPKZV49Tf-ikX2O8uiCb4ccgRur6znwkE7MPrDTXSMHALn10sUcO_w",
		"e": "AQAB"
	}`)
	testKeyRSA_2048_3_Priv = mustLoadJWK(`{
		"kty": "RSA",
		"kid": "d214aea9b78d15dd2667f5d19c36df6647fa50023140fd4f5866d285718b897c",
		"n": "qFnqWpBOZ_5jnIr_XkXnonmHII5gKzxPhUBfvWBhGN2eH8nmnGh6aUnKyKuCLP_qfYLU7cf9bah-e00451iGite8Tg9ZMYAPFX4NM5j0rGWNv9Z6lSn1xXezJv-FU9VUwXm0DG5eVcB8OV99JWxHivQUSBzY0Q3DUlgrD7FhMd7BrrcmTUO07KnW6SxN3oTbT7fNMxfJSTRxmvU-t-6sLhYP4Wbg39zEUgI8M7b7tvHp34klSqOn2DibVBhvWGF1-IgNT7ng6pS5iAZN7Cz7NjPc9ZM_IWyngPKZV49Tf-ikX2O8uiCb4ccgRur6znwkE7MPrDTXSMHALn10sUcO_w",
		"e": "AQAB",
		"d": "cCIT2uarktD6gFaE6cIeGzZfLuwmWiX9wX-zRWxgwDM9E2dj12IvxtmD3E2Ak4CSK69tLEQ9JUFJnc89y7pHQ0uW_VdzzWjCo0omeOu0bO_njpPJanlcXn7wMVWY9NHvdj8eEfmhk_R1ybE0piyNKpyQtcehEv3bz4kyhW1ck936zafn5GWxCEWKIF-7OcotxtOl6z1GrJ7WqglMk8ooLucnAXzaPtcvD6seUhVH0vG60yI2AEInI_jMHR-mfDxrebG2xPPJLsqH3GChqTsxG5vjuaq71sNBiY_TbaUDbmWZxV66zdjhN93Rw-lc3vCPwG5vmuA2igWH7SKsHmQOAQ",
		"p": "xvTQi8nCJGyZQm3fr7HU23FkSUYkKFUN1FRDWqjyz7kFj5rJxq5dSsNrrgVBvIBNY4TwDDcYia4rvx7KSRRwfV_rsU2-TCOxBebqVlbeV0HI7xMcVZXsv61Eugz-dznUU4wqmP_rCQ8vyNcVvCBbBLBW92n-IRQXsfuSUfuHnT8",
		"q": "2J64YbgvH62PB4h-0iYuaykgUzWVSrXENg3_GaF8877AMTe98gjcH7HzTDW1fOhIcNZtmYTwlA2SgFudLsIJLoOn0-kmHzZjZJTWcsZsfTVwKhHD8Lzic7QkrrlbTIZlVu4mfT2f0kvaQiZp6zDCke3SLvVjlD-ebqTFCDIlXkE"
	}`)

	testKeyRSA_4096_0 = mustLoadJWK(`{
		"kty": "RSA",
		"kid": "4941c1e95df518587e54ddde3510c084fedf92098473537fb3fe022eaece9b42",
		"n": "ypJYaH9_PLdEgGN3WFPL1V9qhC2BRXr24f1kGocaQe7wORFlZGX6gRpOCQ1lLmziLafERoSqZuBWF4vSGLeanCRQ7UKObYalMRR_rTJk6D_VD95LNOmtR4DrucdZvjQrAuLXhM5dNLGVoSq_zQowLDMeLWDUBB_TK4WlofoxI0RVoUOt4UXnSev8M_6yIYASpOAk8gG4FAqRfmU-3JhgkZOQCD0BhKYZVw_kEhRBrS3aik-1pQkan9hYIxWwc4KKtrLHNls7--vk5xDk2Vod5sZYRLlimbqHavN2IBeAeJGJqt1grXOVlnagKqmyq2MoPKufwptJBiVrVsTplyUB9FZe9FxUfxrkkxYJHufwP-3wXhaXpPFdL86TCUuOz-jTfng1rsTpwzmwm3RqFzz01yMc1RhaeWininQqj42bhW4bgFGVRP2QqSFbCbRZ4WMW3qWr3aR2QpJ7b8ac7JUhAJoh7-UtMmyGx9UJP0gjA8Wm-O81UENoDjMMdNUaVTfpSUKB7xZppg244jNZDIm1ptq4QyvvWkr0brp9Ymxpu0e3g2HyywyPjbeyNrIelb4E2tU5vCR4Fs_zGFG71AieruJGfzPTXE7ZaiLhP8n1652NCzIH76Y7OGz4ap2D55-RKpUkSI3KqWxUKxkM0tuZ7LS2F-1wEVs8P01K6RuQpxE",
		"e": "AQAB"
	}`)
	testKeyRSA_4096_0_Priv = mustLoadJWK(`{
		"kty": "RSA",
		"kid": "4941c1e95df518587e54ddde3510c084fedf92098473537fb3fe022eaece9b42",
		"n": "ypJYaH9_PLdEgGN3WFPL1V9qhC2BRXr24f1kGocaQe7wORFlZGX6gRpOCQ1lLmziLafERoSqZuBWF4vSGLeanCRQ7UKObYalMRR_rTJk6D_VD95LNOmtR4DrucdZvjQrAuLXhM5dNLGVoSq_zQowLDMeLWDUBB_TK4WlofoxI0RVoUOt4UXnSev8M_6yIYASpOAk8gG4FAqRfmU-3JhgkZOQCD0BhKYZVw_kEhRBrS3aik-1pQkan9hYIxWwc4KKtrLHNls7--vk5xDk2Vod5sZYRLlimbqHavN2IBeAeJGJqt1grXOVlnagKqmyq2MoPKufwptJBiVrVsTplyUB9FZe9FxUfxrkkxYJHufwP-3wXhaXpPFdL86TCUuOz-jTfng1rsTpwzmwm3RqFzz01yMc1RhaeWininQqj42bhW4bgFGVRP2QqSFbCbRZ4WMW3qWr3aR2QpJ7b8ac7JUhAJoh7-UtMmyGx9UJP0gjA8Wm-O81UENoDjMMdNUaVTfpSUKB7xZppg244jNZDIm1ptq4QyvvWkr0brp9Ymxpu0e3g2HyywyPjbeyNrIelb4E2tU5vCR4Fs_zGFG71AieruJGfzPTXE7ZaiLhP8n1652NCzIH76Y7OGz4ap2D55-RKpUkSI3KqWxUKxkM0tuZ7LS2F-1wEVs8P01K6RuQpxE",
		"e": "AQAB",
		"d": "VQagPRxm16FFC260hUqG4ASwvNIs1HEMd0bYYZobl1knU4zNthpnzxCveHU65wWk2ez1IXRF4fB_slpp0R4fszI7FZs-FRLS-4rTHGtul11TnNl9T7RVmxGt38ihDojvFMMKGyBTVu7DE2bSIsoH9kVugTWHSEPjav0pzJcrUNY56vpxXYDt18VJkrlxI0aSjMnYOAwoq6DT-O2eORFsVy5M4mhY3sipEjYFUOFXv8zjUfKrF55-omE4fWF5MsK0XoMjwtkAkHkvFx2sMN72dgsCubXmgQgeFvIhvs6eifzsf99z2NoPC5y3FbEs4Ws5VF3lLNXpDL9gEoeMVHigHKNoztIzbJnFLjzEag4wOBqbig4nWIlYEcLwd5-E0oQ8GHAhWtJn5AJQKOwTcasRWiUnRAulCcI8f9R2ATTSjAGU-juFUfhProp4w3ZlnUbO6U3tbh90KdFSJa-Bapts6ijPgEp6MHubVCHIxh1KGmod_CuGGjze1GuISEwI_4Yh0SrFlp4Wy6ecg7mKNnXvkBMd6h90LJaTzubnts33cCs-5UzCtYqmWHwXrMA6dRJuicY6su3NXxStuZn2bxU_Dn8LrS1vx_lhDU__NqMNbYj5ZvHkPvZiJNBI6Z1R5SY3pQzpH6vh9W2KRKF8cKOkushIaHNnHo7W5o4eZ9HjJFE",
		"p": "7Bxd2wTrXfeSZAEGZEnfoJaMh6hKFG-fx0wLmF9U-Myq1fCIQ-gdl_Atqy0xQxYtQl2lOPUWfySegJSxpdRF8pwvLq9Yt7xFQDrV0B8d5-FjYJ_Mk2bbyHv_PhNmnfcYHqE_VdADeSaU-MRV1iknvjdbxSxYUF-U2C6llfJhn-ZbGPmZJyz8fTWHSh8aKLPJkuYV84wcNL04hpFPJq7Bhy7HAFhRYNpxXyDO3I36DCU94uvUCY758VKLl8IYMMSqjh-thluKJSkWibo85ZOJR38XFksMr70pH-DkT0WMv3PpLtNu6ac6Ry4CEkZexSWmbVk-pduRooo0On1CMsrsLw",
		"q": "26K4gFJ0Ygqr4KaJ4QcRGN29XcjDpX5XTlHy3hCs_wEWc5wedydVgIH4tadiZcu0WinTtl0I6FYewWNpI8nFh8XJXsLOLepdKKR798zW8kEhzxdYdP1MTtGfy12KUCMeS2RQRsBJlmNdgrI0_m0JvW2LlGeK0EqvrqeS06L8bKGXkfY2aFGHl6ipSfUd5OpuSUqgfrj4-HgB8zywm-yAvn7UK-ySyQrP3K4PuOiY5QZmcPDxl8yqDAjorStrjPieYrlMqREHMsyq2FtAXhx4FzpKN7OljbE3EpgyFKkFN-wDTWRGxu3au0hQJx0-XthZ9i5r8zGcEbArQ5jZH7CQvw"
	}`)

	testKeyRSA_4096_1 = mustLoadJWK(`{
		"kty": "RSA",
		"kid": "d01929d205e83facdc5080a0a4ade80853978298a5c73780e863c035d32127c9",
		"n": "ve_rSTi1vwfV2Oj7YNu1kv8Xz2ImAEqf7qo2RehTYW4FJpbj0wkMBVGmAulm4dz-1knnaPdpbrrIwkEYr1XR6kSIeB8aXkCZqDGZATxLSRqGEzu2J8Xt9z_qTQLWYD-NuMEf9H3B1CiP3Q2RnoWUlGvOr_KI5h6anzeqgZxlQLgqQujXscwqWpX7MVh3sheZ5SbyTkDcFWvQS_NEPaBGso-Au9X-yhZsHU1Ky02Nd_DPOrrRzl7uymE07hy3bIbxmrh4ZeEz6P2ixsHHYbd15GNMRlr1cWbg91RBB-akSxX0VYoLjjuqwo33UHk1hBbSATbobfpOKRruuZPZ_xOiPi-m0tUdD1Pj-h6xRcA5sZ1d55IdL8IY_9kgLc_WyU2RWFTXV6zA2SDDdE6EdEB_8q6U0oLU5T-YRxd-V2qOTqW1388qUaL5BalSvqM0g8kVfBYIM3uwLqQI_hCer4CL7_0DGlUtheye3qcoyqWVJd9iqfcWC-1S8NljAJwM9sehx6kOSM85UuUTH89VM35oAVp0rSPcLy20dPzEVQ0LAcy_iRHTkqy9nZ1z6mo-IWQE2ZyPCuESEfG7nFJ3YlUfwBJ5uZnk0N1FVYX5zgEKdkP322vShrv1blB_JtUedpbCcuS-qCaywpwjL0pzTDKpJMEqHds0-DFT7AbULcujw_U",
		"e": "AQAB"
	}`)
	testKeyRSA_4096_1_Priv = mustLoadJWK(`{
		"kty": "RSA",
		"kid": "d01929d205e83facdc5080a0a4ade80853978298a5c73780e863c035d32127c9",
		"n": "ve_rSTi1vwfV2Oj7YNu1kv8Xz2ImAEqf7qo2RehTYW4FJpbj0wkMBVGmAulm4dz-1knnaPdpbrrIwkEYr1XR6kSIeB8aXkCZqDGZATxLSRqGEzu2J8Xt9z_qTQLWYD-NuMEf9H3B1CiP3Q2RnoWUlGvOr_KI5h6anzeqgZxlQLgqQujXscwqWpX7MVh3sheZ5SbyTkDcFWvQS_NEPaBGso-Au9X-yhZsHU1Ky02Nd_DPOrrRzl7uymE07hy3bIbxmrh4ZeEz6P2ixsHHYbd15GNMRlr1cWbg91RBB-akSxX0VYoLjjuqwo33UHk1hBbSATbobfpOKRruuZPZ_xOiPi-m0tUdD1Pj-h6xRcA5sZ1d55IdL8IY_9kgLc_WyU2RWFTXV6zA2SDDdE6EdEB_8q6U0oLU5T-YRxd-V2qOTqW1388qUaL5BalSvqM0g8kVfBYIM3uwLqQI_hCer4CL7_0DGlUtheye3qcoyqWVJd9iqfcWC-1S8NljAJwM9sehx6kOSM85UuUTH89VM35oAVp0rSPcLy20dPzEVQ0LAcy_iRHTkqy9nZ1z6mo-IWQE2ZyPCuESEfG7nFJ3YlUfwBJ5uZnk0N1FVYX5zgEKdkP322vShrv1blB_JtUedpbCcuS-qCaywpwjL0pzTDKpJMEqHds0-DFT7AbULcujw_U",
		"e": "AQAB",
		"d": "mCjZ3wDVaMJIKMsMhx28KpS9aGACfX1K_pHRhNOH6KeQ7Mc4oFnBDYnJas-8ofi_FsCB6G88QX7VUfmAYwZncjuQ8FpKb3NlJX8GSh0ZWukqu8G8PcSszMShWSyKvPRs_rOIe_87BlGwXrB-FfaBfx2WqRGtZlziFecsa0T1QJHJGW0bTs52p7c7Ut7ClSOfIBrBRrtjFK4YYp_x7US3HlkkElZvFUo9NoQzBQeN66Y4_Z2ocqFOv0Z8drz-nKzGZOKfYU62nVKD0qJurfOhOGPsOPipZD28v6b5qfC1cYmXAefjNgDK3a2JkShpHPaDKoHoViKN9xQiZvzxSQ1bjQCgB6xQWk61YwmiB44gFLshXvXkx_wcccO2XC4m-Pk0r-Uj9Ugvgizk8HYp1H9eqnp8A6sqU44mrmdt14vM68SGzEqYpLoUkcxPttHvNoORAzgQfuK3SOIFyuwFqKTMBupR4A42XvxLj_PuKEKFekS2JEK8fON2t_LuUDVemjMxwXD0gjljHSKD9HnguihI5IEd6Kgr-HlTwiiDQfzcmWQdoFUs_Je4tVCSbGRrJYju01gLouDu7gzN14pZcEyoBLPkj-_MDiiymvlBLOBU2s5nQRZCaC_0IKgHLaT_rzQaXwmVhU-OURdwdoViMVc_cdXtle4TqU1g41nNE_0tiwE",
		"p": "2h_9HZCOpf3S9qpAkNS2ncMsRjBlWCpRltoCs-XBF_IgOJlCkJ-42k_V1zaf6YrgMy00XD4Ij5MMiUOharvc3w_WKJF-AwGu9JEpi38Wzj1nkY86sdB1ZPu4ihSRZm81jIjuLZ2siILqkupXaY0S_0CeNLKPoVpvfPWCb3psCWuLVRDhYECPCx9FYg2tzisbFpVAPPqNCFswMYNvhyJ2NQ0hzueKDc1uHBUanCSG6gIbeY1bQvYYFGySYkWZcQg98RSeDbLysHalCSQCgPjss3likTT9OjWLv23vpXURFhgywv6Phx4He0yWB8ZRrnskfW0S76kZ325SPUlnOhf31Q",
		"q": "3urwELOa7TDpwLMeMEOf2aspz1gkMjoN-V4SMzRGxIoPy93-bQ62Z8nPduma1gK0zHOIs0gU8dDqjy8P8Z7vudJqRFJIrzjD-RnDqQDz7KM_6x0kbEN4GLQiNGVAckRhoEmYjDXXkG3teD2ST5NqlYwxGvFmCvw5RCJZG7eNjXY3KnJQrDyx61NIFEvSm3sBlJkOVhSanuqHBL6T-efaBkUcQ7mY7_iYtCt7cjMG8qQtF0KovVsMF6BUc9KDobpUvDsvPrSTAzsYJvVNkzGJtF8Strr-61VA6qZnRcDW3ukYvmJp_SdAcv3KzSkWwEmJEaAtaiOh2jWkRG9QpV_LoQ"
	}`)

	testKeyRSA_4096_2 = mustLoadJWK(`{
		"kty": "RSA",
		"kid": "822512e8098b9fa56c9152a56d2607bda912c89f6f4fe1b6ccf41627bf06de29",
		"n": "3bcg0diA0TPvK283P02o5UT8cBf5q48uUsfuCkIbLqNFtskGuoft8OJLmik7c7pLNSK3Jad_-xRlfsrcV1fDBs8EJDFnoY-4r_xk_-GQl-OsHzuz7HMvjVYUSYeVZMtzPAfCIFJxSIaiIW9MEwCwYyrtRefedfKUV7Ax6hjOUXtyRZ5DRHj2FnNl6MjE8mTSJ8OQI9ETErg0wF448fJb6hYCidVHfeBSCOceljXMg8a9iDYHVnzsUb9ETl7xlRZDvUpIoG0jU-aC3oZdM0YgXspvw4FcXg1-ad-TbgRjEm_iJ-4KUhAYso6-Y6eimDT_TMhbKBLGE86pUjRPIXVNl6odjkXmcmh7bz9v0-gIfHB9zVbrqE8i8iIwkux-Uhq_VMiIFikjtjmDo5W-7qA_cjvq1dA1QqOJ54ijbdvglUtneuA5CnwfGHAvf94lEPebgP5eCMZjz_Dtl0FTX0u5lOVBs4qeyEfV1XANOq_h1gHmwJDlspGZVuPfSk4-YMCeBy5Ua_gNDqOIQo3EQbmjyN5yD9byFh3WDSQJN1kKT3BZSou-Q8-KT2lu9KT_CpTfVMaNXnPpXJ_H0S9Q7sbUzA6dF1h0Q_mu_bOlWTRlH3vXfZ3_3Vikjk-jhQdaRv8yjTdhX33wAVHk7omKlo76G3QJdnvjysJ6aagFPZ3qhmc",
		"e": "AQAB"
	}`)
	testKeyRSA_4096_2_Priv = mustLoadJWK(`{
		"kty": "RSA",
		"kid": "822512e8098b9fa56c9152a56d2607bda912c89f6f4fe1b6ccf41627bf06de29",
		"n": "3bcg0diA0TPvK283P02o5UT8cBf5q48uUsfuCkIbLqNFtskGuoft8OJLmik7c7pLNSK3Jad_-xRlfsrcV1fDBs8EJDFnoY-4r_xk_-GQl-OsHzuz7HMvjVYUSYeVZMtzPAfCIFJxSIaiIW9MEwCwYyrtRefedfKUV7Ax6hjOUXtyRZ5DRHj2FnNl6MjE8mTSJ8OQI9ETErg0wF448fJb6hYCidVHfeBSCOceljXMg8a9iDYHVnzsUb9ETl7xlRZDvUpIoG0jU-aC3oZdM0YgXspvw4FcXg1-ad-TbgRjEm_iJ-4KUhAYso6-Y6eimDT_TMhbKBLGE86pUjRPIXVNl6odjkXmcmh7bz9v0-gIfHB9zVbrqE8i8iIwkux-Uhq_VMiIFikjtjmDo5W-7qA_cjvq1dA1QqOJ54ijbdvglUtneuA5CnwfGHAvf94lEPebgP5eCMZjz_Dtl0FTX0u5lOVBs4qeyEfV1XANOq_h1gHmwJDlspGZVuPfSk4-YMCeBy5Ua_gNDqOIQo3EQbmjyN5yD9byFh3WDSQJN1kKT3BZSou-Q8-KT2lu9KT_CpTfVMaNXnPpXJ_H0S9Q7sbUzA6dF1h0Q_mu_bOlWTRlH3vXfZ3_3Vikjk-jhQdaRv8yjTdhX33wAVHk7omKlo76G3QJdnvjysJ6aagFPZ3qhmc",
		"e": "AQAB",
		"d": "wcO4mAxJUAu-OsxgkR9SusPWhjQ9y5Q_XLNDso1hahng5ES9b6k55mouvlTIk3Q9I_vp6auAKrMBnJS3ilG1rK6hJOxUcBrFwm-m6QV9s3CSzV0E-mEULsYKxtQKWOOBGvaAznSeck7PRL8a0gSpIpGyeYSRo6zTverLRJZXQVjMXlFY4m-ASdCiQJWtoVVBYOUFhHfE3nhECdaOl8xCTcrcfw75AuZXa1ZpIcd0q7m1jGQDd6-HbE3m6UMKiEvD-ZsA68tVs45h0w3ER_pCcfUjRc45Ji1OzEJLezu0RbmoAVOEi4FrxCkB9N_dNn4inD0BhX0axNa4nZH_kfMNUh06081FaBiBq5bNBPQF7lWIYxAhtd2VgsUTZtNxMfHDTllt5fvbCogYngU5-Wj3UM33R_es_aakt_CFndJPmyenP0J7KEBmf4qnlrbxun-jQNV_cbEzgQA_Pt5RIhB5jgofYzT72Ib-lViZUolXpxg8mFQ5BPghqriuaX8eSc85y5rbu-nEvheDkIC7xjTtDBShXMvKKokx6t4d-x_W4sWDuLKsVNn3Veg6mbbl3O74Fr-joEM9unMg-9KCapAuBKmMv4GejOVOIj43i0lm8dsuMslNKsThWpjApE50-VBuySi1f0jD2HNEzjMTt5ZrlNi9bRFzAht46kdc_g4TWLE",
		"p": "7cl5QebQldiOrfe2Z7E2aP4d1GIuX4b3u-kXUDa73HI4S2u9TooJF8lfsbrtzJIG60iIzio9Pej8R4ss2_1TPC_F7gQS67ST5yY2zuwUH6jwpt8KtVO3NKd9SNfVJse_mZLp7eAeH8CMq4bqkDig8ZI_Sp1LQtWXvhvhju40i0h1ef-MD36wbc26BE1L6tQMAoJIU-pMDawmWMGG3Q5_LxKHb1cxvV8gVfFgsJbAHMPXxcmpCdp1FqeqpvRs3bch3AMxHwjy1SFRK6lhvliS2ZvD7fdupA09PGJ-cNO3S9iHdCvsnyOHg9pMbxxcYW8Io_t1ihk6JYFvZlDIYvukXw",
		"q": "7rKFf6190MlunvlLSdMvN_E2Xuow35_EThaxQ9e_7FS32LKPy847vFreXzVivVW1IVXrRSDgU8VIEyBQzftliLSriefKYdx2jtd_A84baha4p6CAK9SwfZcVEcIuU2Ul8XcUpbg53_bxUWDv4mYewZW_AqaZ0ZR1Ug14gCGHELvjs7MmWkCkJB6lEenRQZvzwcmA8tWzZ6Dlqb_RyLtxcVPbeSuc3t6YxbB1-pquwXoQLB9-XvgR_4L6vuGZrp1D-C0lzboUPoftkC_hKYb_xY6LPJTsL8LUypATcTrAnD0TBrqj5-Oy_4dMin6OcsXrfdxflxaJqkh7B-kz20oa-Q"
	}`)

	testKeyRSA_4096_3 = mustLoadJWK(`{
		"kty": "RSA",
		"kid": "ad60df152305985e6bdd8cb1d1f496f766aed097b7464b8b821a09b27f054dcb",
		"n": "q8mHub9-A7LH2ykklO52CRQ9KYeM9FlHaoLJCMokmwdb8FF8cAMOPDQ4lx7rB107vwX5oHS6T4uhOUH6ppgZnvB083-f5cBfvZPSwvKg1csOPI3Llp-hIO3vnu7v4M-dXRJevmNdTAoHINXP8mJ1KfGL6BuxckxLEOoIX0suSmyudAE_lc6ioMAfwehkH7qTWwllIuReGwxHHYapM0AFIYysJ0z96l-WNkO3ssLm6Q7txpumM_b0d58OkdezBLujEfn0t5H8hsml_LUSE4TfWcGwm3oDBB2q5tsI2_QmVwdV6BcP43Sm8R7EbJvTQq3YXKLlC-Et4-3CDGaMV6z1eFmHBs_FMGzto-kg9MUFockOe0vkfjHj6AafNw7AFyBJRvV3EdMM-O1ysiC5cs45aowbfQIz_9H9RVbR9iMiEjq6nM34bhgfu-FvOmAcliCOJnDlBy2wWe2XyXTLQVkmn1C_w2mitcQAkP-LiktypUwSIbyDqfBai1_imO1SIVzomRLvzSju5qGrcm0ROV6o9zKXCVc546g9la9FUTTPZsI0_s3hcVRVMaF3fa122hE848serjCqNJ0Nb3CYOlVCc6JrkUdeuCj0Y-on6V6aeCFAQxFb1z_p88STsNJkrcNZp87f1JUaSB4POF01dLNQ2HTe7xJCMRmK-fpNuDCOzNM",
		"e": "AQAB"
	}`)
	testKeyRSA_4096_3_Priv = mustLoadJWK(`{
		"kty": "RSA",
		"kid": "ad60df152305985e6bdd8cb1d1f496f766aed097b7464b8b821a09b27f054dcb",
		"n": "q8mHub9-A7LH2ykklO52CRQ9KYeM9FlHaoLJCMokmwdb8FF8cAMOPDQ4lx7rB107vwX5oHS6T4uhOUH6ppgZnvB083-f5cBfvZPSwvKg1csOPI3Llp-hIO3vnu7v4M-dXRJevmNdTAoHINXP8mJ1KfGL6BuxckxLEOoIX0suSmyudAE_lc6ioMAfwehkH7qTWwllIuReGwxHHYapM0AFIYysJ0z96l-WNkO3ssLm6Q7txpumM_b0d58OkdezBLujEfn0t5H8hsml_LUSE4TfWcGwm3oDBB2q5tsI2_QmVwdV6BcP43Sm8R7EbJvTQq3YXKLlC-Et4-3CDGaMV6z1eFmHBs_FMGzto-kg9MUFockOe0vkfjHj6AafNw7AFyBJRvV3EdMM-O1ysiC5cs45aowbfQIz_9H9RVbR9iMiEjq6nM34bhgfu-FvOmAcliCOJnDlBy2wWe2XyXTLQVkmn1C_w2mitcQAkP-LiktypUwSIbyDqfBai1_imO1SIVzomRLvzSju5qGrcm0ROV6o9zKXCVc546g9la9FUTTPZsI0_s3hcVRVMaF3fa122hE848serjCqNJ0Nb3CYOlVCc6JrkUdeuCj0Y-on6V6aeCFAQxFb1z_p88STsNJkrcNZp87f1JUaSB4POF01dLNQ2HTe7xJCMRmK-fpNuDCOzNM",
		"e": "AQAB",
		"d": "QFcw8I8aQYRaemlEfEt8BhaAeed9EZ_GscveQ96CK1ZsRuweMU3TrRTaBS_dU1rGH9u7DS_rABQKBIoDuRXKss7Y3sJ0Pvb4ZObSz5VUS_7LjD6HfBi5nr2_O8W-LnNUOyHAPoq0zOAMn221ftEFlPoVLpAAvBB7JRCiph5gbhuak3RMPm2wV4jd3CCQL5oPys8QBCuIW5UTpalkAf_-a_xmFiouB_RZLGXcjaWWGsAuqm5tp5TdJ1h5eoJRWHp2ryrxTzfsXwdzldyzsn_Xr6Rt4y2lp4r9EY4EGW2uVnY25MCOgOCWDkU5yHvselLmcHvKUdK6_11zinV2Jvhuz1FaW10P6lRXxhjuuT4bT7XmY6WkJCUvWf8w_NYrGwDbRvQa6YZcZZWKZ1l2Enkgd_P4qwtrnROQSCe0dJdNG3lSq90lrAwuvb8AhKhpQ8nJSaSQc_h2pa4NJZ-R__9m0_7CRUuEEg9k47AtgdIe09KLSpcACc3W5cBXEw2pL8ihqcf9AVgM0TFQQVrUdIst3YjUiB6r2zLVCRx8KjtT8Pmz6YtMQwDIFTrUopwwai5PEP3QEdxdNF39W1iwkqjA8uw4IlXgZAr4-9s3x617XUL0BQ4TgMpTsEghpV1U8U2HQfYGUKBcAlSqIlAi0MU4QRwcsAJrcoTIi-e34NFMbHE",
		"p": "xvZ1BKzaRfrmQT2mmbK1MzKdr4amHrx4EJ_4fRsCRBj8MqB92y4Bp0iuO8rZ3iK_yOOgQQTcr4UP5UEACLOqGDPNa92UUpHu1U5XR3MiIY0kGcdovpkkGU8fq78VRpyofc1b9kvZsy4eL0e7W7jApjGu479D_evnpT7lqSOGRl1Z0QHdI3ctKmKw48-AmQWL4BibXTyaXOfRgTHm4AtbrvFshwCZ43QgzrCrwbhZlOfiIIW9wXa_OqqOhV11BLF72qTglecfTgUOxY0W5GWgbwXCVZ8Lwr6cxYKGrKjmny9UNOIjABGsJjZm5egwDNUhhoaEeXXyzXbCKynvUCDxZw",
		"q": "3Qi1bRzj3TYW3Iw9KSDCuuHWBJRO_3Ke-JLB7PHEXc1hsPwF_XDqXPaBbIUtgKaqNzUehYAQbtHHhJVGqZadTKjsNkO4sO_r5nRoxUpFuGgUEmKNN0QeJa_llHzAZ9JpvUrtoyzVZX-2Em_3aAhtV5pZ6t22YToPDjZIBgqL96MQRCyKsv2FS_dbpoYKdXlG6phzkzKPn3oaWkh-VwmgAsFg7uXdiquQEsXYcOq3-77Umiuke6SBbaHLryLflTzOV7YvY7Kw3s3NycS9MDBESKYXqqX3YKvS25ZFFjYbrVOx5AluLiwajZu2biIPZb9rNbSXE77Hll3tAbmA5syJtQ"
	}`)
)
