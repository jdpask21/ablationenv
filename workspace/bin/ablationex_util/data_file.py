seedlist = {
	"totinfo" : {
		"v1" : [300, 320, 340, 360, 380, 400, 440, 460, 480, 500],
		"v4" : [n for n in range(300, 500, 20)],
		"v7" : [n for n in range(210, 400, 20)],
		"v8" : [210, 220, 240, 250, 260, 280, 290, 310, 390, 400],
		"v11" : [n for n in range(300, 500, 20)],
		"v12" : [230, 250, 270, 290, 310, 330, 350, 370, 390, 410],
		"v13" : [n for n in range(300, 500, 20)],
		"v15" : [n for n in range(300, 500, 20)],
		"v17" : [230, 250, 270, 290, 310, 330, 350, 370, 390, 410],
		"v18" : [300, 320, 340, 360, 380, 400, 440, 480, 500, 520],
		"v20" : [230, 240, 250, 270, 290, 310, 330, 350, 360, 370],
		"v23" : [n for n in range(230, 420, 20)]
		},
	"printtoken" : {
		"v2" : [20, 30, 60, 80, 100, 120, 140, 160, 180, 200],
		"v5" : [220, 230, 250, 260, 270, 290, 300, 310, 320, 340],
		"v7" : [220, 230, 240, 260, 280, 300, 320, 340, 360, 380]
		},
	"printtokens2" : {
		"v1" : [20, 60, 80, 100, 120, 140, 160, 180, 200, 220],
		"v2" : [n for n in range(20, 210, 20)],
		"v4" : [40, 50, 70, 80, 100, 120, 130, 140, 180, 200],
		"v5" : [n for n in range(20, 210, 20)],
		"v6" : [n for n in range(20, 210, 20)],
		"v7" : [40, 60, 100, 120, 140, 160, 180, 220, 240, 260],
		"v8" : [n for n in range(20, 210, 20)]
		}
	}

functionnamelist = {
	"totinfo" : {
		"v1" : ["InfoTbl", "QGamma", "gser", "QChiSq"],
		"v4" : ["gcf", "LGamma", "QGamma", "QChiSq", "InfoTbl"],
		"v7" : ["QGamma", "gser", "QChiSq", "InfoTbl"],
		"v8" : ["LGamma", "gser", "gcf", "QGamma", "QChiSq", "InfoTbl"],
		"v11" : ["InfoTbl", "gcf", "QGamma", "gser", "QChiSq", "LGamma"],
		"v12" : ["LGamma", "gser", "QGamma", "QChiSq", "InfoTbl", "gcf"],
		"v13" : ["InfoTbl", "gser", "LGamma", "QGamma", "QChiSq"],
		"v15" : ["InfoTbl", "QGamma", "gser", "QChiSq", "gcf", "LGamma"],
		"v17" : ["gcf", "LGamma", "QGamma", "QChiSq", "InfoTbl"],
		"v18" : ["InfoTbl", "gser", "LGamma", "QGamma", "QChiSq"],
		"v20" : ["LGamma", "gser", "gcf", "QGamma", "QChiSq", "InfoTbl"],
		"v23" : ["InfoTbl", "gcf", "LGamma", "QGamma", "QChiSq"]
		},
	"printtoken" : {
		"v2" : ["get_token", "unget_char", "get_char", "get_actual_token", "strcpy"],
		"v5" : ["get_token", "get_char", "constant", "get_actual_token"],
		"v7" : ["get_token", "numeric_case", "check_delimiter", "get_char", "get_actual_token", "unget_char"]
		},
	"printtokens2" : {
		"v1" : ["token_type", "is_spec_symbol", "is_identifier", "is_num_constant", "is_str_constant", "is_char_constant", "is_comment", "is_eof_token", "print_token", "get_token", "get_char"],
		"v2" : ["get_token", "is_eof_token", "is_spec_symbol", "strcmp", "is_token_end", "is_keyword", "token_type", "is_identifier", "is_num_constant", "is_str_constant", "is_char_constant", "is_comment", "print_token", "get_char"],
		"v4" : ["is_keyword", "token_type", "is_spec_symbol", "is_identifier", "is_num_constant", "is_str_constant", "is_char_constant", "is_comment", "print_token", "get_token", "get_char", "is_eof_token"],
		"v5" : ["is_keyword", "token_type", "is_spec_symbol", "is_identifier", "is_num_constant", "is_str_constant", "print_token", "get_token", "get_char", "getc", "is_eof_token"],
		"v6" : ["is_keyword", "token_type", "is_spec_symbol", "is_identifier", "is_num_constant", "is_str_constant", "is_char_constant", "is_comment", "is_eof_token", "print_token"],
		"v7" : ["get_token", "is_spec_symbol", "is_token_end", "is_eof_token", "is_keyword", "token_type", "is_identifier", "is_num_constant", "is_str_constant", "is_char_constant", "is_comment", "is_eof_token", "print_token", "get_char"],
		"v8" : ["token_type", "is_comment", "is_eof_token", "print_token", "get_token", "get_char", "is_spec_symbol", "is_token_end"]
		}
	}

faulty_function_name_list = {
	"totinfo" : {
		"v1" : "InfoTbl",
		"v4" : "gcf",
		"v7" : "InfoTbl",
		"v8" : "gser",
		"v11" : "gser",
		"v12" : "LGamma",
		"v13" : "InfoTbl",
		"v15" : "gser",
		"v17" : "gcf",
		"v18" : "InfoTbl",
		"v20" : "InfoTbl",
		"v23" : "gcf"
	},
	"printtoken" : {
		"v2" : "get_token",
		"v5" : "get_token",
		"v7" : "numeric_case"
	},
	"printtokens2" : {
		"v1" : "get_token",
		"v2" : "get_token",
		"v4" : "get_token",
		"v5" : "is_str_constant",
		"v6" : "is_num_constant",
		"v7" : "is_token_end",
		"v8" : "is_token_end"
	}
}
