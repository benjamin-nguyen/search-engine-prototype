from invert import build_invert

def main():
    build_invert("out_stop_stem", use_stop_words = True, use_stemming = True)
    build_invert("out_nostop_stem", use_stop_words = False, use_stemming = True)
    build_invert("out_stop_nostem", use_stop_words = True, use_stemming = False)
    build_invert("out_nostop_nostem", use_stop_words = False, use_stemming = False)

if __name__ == "__main__":
    main()