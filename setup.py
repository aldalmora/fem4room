import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fem4room",
    version="0.0.1.dev4",
    author="André Luiz Dalmora",
    author_email="andre.dalmora@gmail.com",
    description="Room Acoustics Simulation. Retrieve room impulse response using finite elements method.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aldalmora/fem4room",
    packages=setuptools.find_packages(),
    setup_requires=['wheel','numpy'],
    install_requires=[
        'scipy',
        'gmsh-sdk==4.4.1.post1',
        'matplotlib~=3.2.1',
        'ezdxf~=0.13.1',
        'scikit-umfpack==0.3.1'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Development Status :: 4 - Beta"
    ],
    python_requires='~=3.6',
    keywords="room acoustics finite elements method impulse response",
)