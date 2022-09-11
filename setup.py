import setuptools

with open("README.org", "r", encoding="utf-8") as fh:
    long_description = fh.read()

install_requires=[
    "torch>=1.8.0",
    "transformers>=4.4.1",
    "nltk>=3.5",
    "spacy>=3.0",
    "Flask==1.1.2",
]

setuptools.setup(
    name="clinisift-samrawal", # Replace with your own username
    version="0.0.2",
    author="Sam Rawal",
    author_email="scrawal2@illinois.edu",
    description="An NLP tool for parsing, analyzing, and visualizing medical records",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/clinisift/clinisift",
    project_urls={
        "Bug Tracker": "https://github.com/clinisift/clinisift/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=install_requires,
)

