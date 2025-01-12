from distutils.core import setup, Extension


def main():
    setup(
        name="c_neurite",
        version="0.1.0",
        description="C extension for neurite",
        author="Juan Vanegas",
        author_email="juan@vanegas.com",
        ext_modules=[
            Extension(
                "c_neurite", ["neurite/neuritemodule.c", "neurite/neural_network.c"]
            )
        ],
    )


if __name__ == "__main__":
    main()
