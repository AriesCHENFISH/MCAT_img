# main.py
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python main.py [pretrain|extract]")
        exit(1)

    mode = sys.argv[1].lower()

    if mode == 'pretrain_ct':
        import mae_ct_pretrain
    elif mode == 'extract_ct':
        import feature_ct_extract
    elif mode == 'pretrain_dsa':
        import mae_dsa_pretrain
    elif mode == 'extract_dsa':
        import feature_dsa_extract

    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python main.py [pretrain|extract]")
