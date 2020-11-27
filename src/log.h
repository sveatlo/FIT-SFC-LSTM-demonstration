#ifndef LOG_H
#define LOG_H

#include <netinet/icmp6.h>
#include <netinet/ip6.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <netinet/udp.h>
#include <bitset>
#include <string>

using namespace std;

/**
 * Helper class for better loggin messages
 */
class Log {
public:
    /**
     * Debug log
     * @param msg message to log
     * @param end string to end the message wih
     */
    static void debug(string msg, string end = "\n");

    /**
     * Info log
     * @param msg message to log
     * @param end string to end the message wih
     */
    static void info(string msg, string end = "\n");

    /**
     * Warning log
     * @param msg message to log
     * @param end string to end the message wih
     */
    static void warn(string msg, string end = "\n");

    /**
     * Error log
     * @param msg message to log
     * @param end string to end the message wih
     */
    static void error(string msg, string end = "\n");
};

#endif
