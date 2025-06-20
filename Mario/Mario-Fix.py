with open("Super Mario Bros. (World).nes", "rb") as f:
    header = f.read(16)
    if header[0:4] == b"NES\x1a":
        print("Valid iNES header")
        print("PRG ROM size (KB):", header[4] * 16)
        print("CHR ROM size (KB):", header[5] * 8)
    else:
        print("Invalid iNES header")
