
DEBUG=-g

CC=gcc
CFLAGS=-Wall  $(DEBUG) -I.. `gsl-config --cflags`
LDFLAGS=-Wall $(DEBUG) `gsl-config --libs` 

QP_OBJS = ../libqp.a

HDRS = check.h
SRCS = check.c
OBJS = $(SRCS:.c=.o)

PROGS = random_qp solve_qp

default: $(PROGS)

solve_qp: $(OBJS) $(TEST_OBJS) solve_qp.o
	$(CC) -o $@ $(LDFLAGS) $(OBJS) $(QP_OBJS) solve_qp.o

random_qp: $(OBJS) $(TEST_OBJS) random_qp.o
	$(CC) -o $@ $(LDFLAGS) $(OBJS) $(QP_OBJS) random_qp.o 

.c.o:
	$(CC) $(CFLAGS) -c -o $@ $*.c

clean:
	rm -f $(OBJS) $(PROGS) random_qp.o solve_qp.o
