
DEBUG=-g

CC=gcc
CFLAGS=-Wall -O2 $(DEBUG) `gsl-config --cflags`
LDFLAGS=-Wall $(DEBUG) `gsl-config --libs`

HDRS = qp.h gsl_qp.h
SRCS = qp.c gsl_qp.c
OBJS = $(SRCS:.c=.o)
LIBS = libqp.a

all: libs

default: qp.o

.c.o:
	$(CC) $(CFLAGS) -c -o $@ $*.c

clean:
	rm -f $(OBJS) $(LIBS)

check: qp.o
	make -C test clean
	make -C test
	./test/random_qp

libs: $(LIBS)

libqp.a: $(OBJS)
	ar rcs libqp.a $(OBJS)
