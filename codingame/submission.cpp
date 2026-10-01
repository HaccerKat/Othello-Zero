#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <math.h>
#include <random>

#pragma GCC optimize ("O3")

std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
int rnd(int l, int r) {
    if(l > r) std::swap(l, r);
    return std::uniform_int_distribution<int>(l, r)(rng);
}

class Experimental_Board {
    char grid[8][8];
    bool player, found_next_moves = 0, skip_turn = 0;
    int game_ends = -1;
    float eval = -2.0;
    int next_move = -1;
public:
    const float BLACK_WINS = 1, WHITE_WINS = -1, DRAW = 0.0;
    std::vector<std::pair<Experimental_Board*, std::pair<int, int>>> next_boards;
    ~Experimental_Board();
    Experimental_Board(char gr[8][8], bool p);
    void print() const;
    char get_pos(int x, int y) const;
    void change_player() {player ^= 1;}
    bool has_no_move() const {return skip_turn;}
    bool get_player() const {return player;}
    std::string get_player_string() const;
    std::pair<int, int> get_points() const;
    int get_sum_points() const;
    int get_winner_num() const;
    std::string get_winner() const;
    void get_static_eval();
    float get_eval();
    void change_eval(float new_eval);
    void find_next_boards();
    bool find_if_game_ends();
    Experimental_Board* advance_move(int input_x, int input_y);
    std::string get_board_string();
};

Experimental_Board::~Experimental_Board() {
    for (auto [child, pair] : next_boards) {
        delete child;        // Recursively delete children
    }
}

// 0 -> black, 1 -> white
Experimental_Board::Experimental_Board(char gr[8][8], bool p) {
    memcpy(grid, gr, sizeof(grid));
    player = p;
}

void Experimental_Board::print() const {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            std::cout << grid[i][j] << " ";
        }

        std::cout << "\n";
    }
}

char Experimental_Board::get_pos(int x, int y) const {
    return grid[x][y];
}

std::string Experimental_Board::get_player_string() const {
    return player ? "White" : "Black";
}

std::pair<int, int> Experimental_Board::get_points() const {
    int black_points = 0;
    int white_points = 0;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if (grid[i][j] == '0') {
                black_points++;
            }
            if (grid[i][j] == '1') {
                white_points++;
            }
        }
    }

    return {black_points, white_points};
}

int Experimental_Board::get_sum_points() const {
    auto [black_points, white_points] = get_points();
    return black_points + white_points;
}

int Experimental_Board::get_winner_num() const {
    auto [black_points, white_points] = get_points();
    if (black_points == white_points) return 1;
    return black_points > white_points ? 0 : 2;
}

std::string Experimental_Board::get_winner() const {
    int winner = get_winner_num();
    if (winner == 1) {
        return "Draw";
    }

    return winner == 0 ? "Black Wins!" : "White Wins!";
}

const int WEIGHTS_SZ = 42016, BIASES_SZ = 321, RES_SZ = 449, RES_SZ_2 = 321;
constexpr int CNT_LAYERS = 5;
constexpr int LAYERS[] = {128, 256, 32, 32, 1};
float WEIGHTS[WEIGHTS_SZ], BIASES[BIASES_SZ];
int16_t QUANT_WEIGHTS[WEIGHTS_SZ], QUANT_BIASES[BIASES_SZ], QUANT_WEIGHTS_LAYER_0[LAYERS[0] * LAYERS[1]];
const float BOUND = 0.05, QUANT_MULT = 46.5 / BOUND;
void Experimental_Board::get_static_eval() {
    if (find_if_game_ends()) {
        auto [black_points, white_points] = get_points();
        if (black_points == white_points) {
            eval = DRAW;
        }

        eval = black_points > white_points ? BLACK_WINS : WHITE_WINS;
        return;
    }

    int nnue_layer[128];
    memset(nnue_layer, 0, sizeof(nnue_layer));
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            int pos = i * 8 + j;
            if (grid[i][j] != '.') {
                if (grid[i][j] - '0' == player) nnue_layer[pos] = 1;
                else nnue_layer[pos + 64] = 1;
            }
        }
    }

    // constant optimize later
    auto start = std::chrono::steady_clock::now();
    int16_t res_int[RES_SZ];
    for (int i = 0; i < LAYERS[0]; i++) {
        res_int[i] = round(nnue_layer[i] * QUANT_MULT);
    }

    int idx_weights = 0, idx_biases = 0, idx_res = LAYERS[0];
    for (int i = 1; i < CNT_LAYERS; i++) {
        // index of the start of the previous layer
        int start_idx = idx_res - LAYERS[i - 1];
        for (int j = 0; j < LAYERS[i]; j++) {
            int acc = 0;
            for (int k = 0; k < LAYERS[i - 1]; k++) {
                acc += QUANT_WEIGHTS[idx_weights + k] * res_int[start_idx + k];
            }

            res_int[idx_res] = round(acc / QUANT_MULT) + QUANT_BIASES[idx_biases];
            if (i + 1 == CNT_LAYERS) {
                float val = res_int[idx_res] / QUANT_MULT;
                eval = tanh(val) * (player == 1 ? -1 : 1);
            }

            // ReLU
            else {
                res_int[idx_res] = std::max((int16_t)0, res_int[idx_res]);
            }

            idx_weights += LAYERS[i - 1], idx_biases++, idx_res++;
        }
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
}

float Experimental_Board::get_eval() {
    // static eval has not been calculated yet
    if (eval == -2.0) {
        get_static_eval();
    }

    return eval;
}

void Experimental_Board::change_eval(float new_eval) {
    eval = new_eval;
}

void Experimental_Board::find_next_boards() {
    auto start = std::chrono::steady_clock::now();
    if (found_next_moves) {
        return;
    }

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if (grid[i][j] != '.') {
                continue;
            }

            char next_grid[8][8];
            memcpy(next_grid, grid, sizeof(next_grid));
            next_grid[i][j] = player + '0';
            bool can_flip = 0;
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    if (dx == 0 && dy == 0) {
                        continue;
                    }

                    int x = i + dx, y = j + dy;
                    int flip_number = 0;
                    while (x >= 0 && x < 8 && y >= 0 && y < 8) {
                        if (grid[x][y] == '.') {
                            break;
                        }

                        int cell = grid[x][y] - '0';
                        if (cell == player) {
                            if (flip_number > 0) {
                                can_flip = 1;
                                x = i + dx, y = j + dy;
                                while (true) {
                                    cell = grid[x][y] - '0';
                                    if (cell == player) {
                                        break;
                                    }

                                    else {
                                        next_grid[x][y] = player + '0';
                                    }

                                    x += dx, y += dy;
                                }
                            }

                            break;
                        }

                        flip_number++, x += dx, y += dy;
                    }
                }
            }

            if (can_flip) {
                Experimental_Board* next_board = new Experimental_Board(next_grid, player ^ 1);
                next_boards.push_back({next_board, {i, j}});
            }
        }
    }

    found_next_moves = 1;
    if (next_boards.empty()) {
        skip_turn = 1;
        Experimental_Board* next_board = new Experimental_Board(grid, player ^ 1);
        next_boards.push_back({next_board, {-1, -1}});
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
}

bool Experimental_Board::find_if_game_ends() {
    find_next_boards();
    if (game_ends != -1) {
        return game_ends;
    }

    if (skip_turn) {
        Experimental_Board* other_player_board = next_boards[0].first;
        other_player_board->find_next_boards();
        game_ends = other_player_board->has_no_move();
        return game_ends;
    }

    else {
        game_ends = 0;
        return game_ends;
    }
}

Experimental_Board* Experimental_Board::advance_move(int input_x, int input_y) {
    find_next_boards();
    for (auto game_state : next_boards) {
        auto [x, y] = game_state.second;
        if (input_x == x && input_y == y) {
            return game_state.first;
        }
    }

    return nullptr;
}

std::string Experimental_Board::get_board_string() {
    std::string board;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            char ch = grid[i][j];
            // nullptr means illegal move
            if (advance_move(i, j) != nullptr) {
                ch = '*';
            }

            board.push_back(ch);
        }
    }

    return board;
}

template <typename U, typename T>
void minimax(U position, int depth, T alpha, T beta, auto start, double response_time) {
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    if (depth == 0 || position->find_if_game_ends() || elapsed.count() > response_time) {
        return;
    }

    position->find_next_boards();
    std::sort(position->next_boards.begin(), position->next_boards.end(), [&](std::pair<U, std::pair<int, int>> a, std::pair<U, std::pair<int, int>> b) {
        if (!position->get_player()) {
            return a.first->get_eval() > b.first->get_eval();
        }

        else {
            return a.first->get_eval() < b.first->get_eval();
        }
    });

    // black to move
    if (!position->get_player()) {
        position->change_eval(position->WHITE_WINS);
        for (auto [child, pair] : position->next_boards) {
            minimax(child, depth - 1, alpha, beta, start, response_time);
            position->change_eval(std::max(position->get_eval(), child->get_eval()));
            alpha = std::max(alpha, child->get_eval());
            if (beta <= alpha) {
                break;
            }
        }
    }

    // white to move
    else {
        position->change_eval(position->BLACK_WINS);
        for (auto [child, pair] : position->next_boards) {
            minimax(child, depth - 1, alpha, beta, start, response_time);
            position->change_eval(std::min(position->get_eval(), child->get_eval()));
            beta = std::min(beta, child->get_eval());
            if (beta <= alpha) {
                break;
            }
        }
    }
}

template <typename U, typename T>
std::pair<int, int> get_best_move(U position, double response_time, bool probabilities, bool dbg) {
    auto start = std::chrono::steady_clock::now();
    position->find_next_boards();
    int depth = 6;
    T eval = position->DRAW;
    std::pair<int, int> move = {-1, 0};
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < response_time && depth < 64) {
        T best_eval = position->get_player() ? position->BLACK_WINS : position->WHITE_WINS;
        for (auto [child, pair] : position->next_boards) {
            // black to move
            if (!position->get_player()) {
                if (child->get_eval() >= best_eval) {
                    best_eval = child->get_eval(), move = pair;
                }
            }

            // white to move
            else {
                if (child->get_eval() <= best_eval) {
                    best_eval = child->get_eval(), move = pair;
                }
            } 
        }
        
        minimax(position, depth, position->WHITE_WINS, position->BLACK_WINS, start, response_time);
        depth++, eval = best_eval;
    }

    if (dbg) {
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cerr << "Depth: " << depth << std::endl;
        std::cerr << "Eval: " << eval << std::endl;
        std::cerr << "Elapsed Time: " << elapsed.count() << std::endl;
    }

    return move;
}

std::string weights_string = "`;SQI<8~P9D1EDLP4~Om[K2@FV4sX[OYa6T]I[;\\t]S@B9@FvV>\'Qg\\N]~OR~MJL~Pb$~ND56\'~MO~MC~NFHFKR1~Oj~M`kYBV]T|~MwkR>P;\\]ZTWKofDAQ_[?eiULraNPDdE7~N?~N&W9hnA~N<~Qd~Or099K0~O|~K8,SV@>gA`Mh^LUROYH\\RSH6aRETkPEI;eVVa730~NBgQWHbA~L5~Kt?HH_.%~Lv~M$!oDVWeN<oDK`KdJbt\\_`V[3ZWW_RIKACXWXqB:HKIbRYAV~OXOh@@F[~OS~O`;7Mn0(Tt~Li~O|R>L]_U~Of\\~OgL~NL7@B`=g~N7B+8SfP?#>Md_ynj~N<~NFAT\\~Od~Ob!~NB8OVRm_~Mj~N3<]BFXO#j~O\\DU~NDbn9~PBT{R~NGc~OSw10~NK\'Kiq@~N<HECFZ>,tUOhZ_~OhG#=LEF*C!Ws<Ls5E_=5%~OWT$E4~P5~NLzL~OQ1L~OU<vR~N*]~MN~O[e2JJ[(~PE~PrW~NC=A.E~P`~P+\\E!e2~NL~P$~OeI3QiWGk~PAB1+SUt|~P\'\"<<GYBg~O[E58$380~OkYE84%~N<r~Ox?d~OP,61j~Q\"W~N=@~NBEP~Ok~OtQ^~NO^&D~MvREFWSKA$~NKlCBUO_2c7JZ5I:~NF~NJH?H9A6~On]~N,:QO)~N:~OX>~QLjuuRQK/~Kvlgae`q}~MS~NOk_qBjM~M_*~N$][oc\\i3~NNaMT\\8~M|2K_~OZEbGHUIKW~Og2Ey,~NF8;HoP~NG{XmDTUT~MY4Yhz=sS$o-NbIa@_/oRUTSSF~NN5PSRQHw~OpFYNJTP~MgZ:Ql|\\P~N$l~NON;QT99\"~OgRe~Omi\"@K\\RQ~NG~N>[T5Ldg$~NE:PgRFc}@~NK/LACTxTHEA-b_IK~OYoF)6OSbYtg<\"E[4I~NJ~N0&d2~Ol~Oy~OPakTx`E~Of~Ofs~NEG~N=Wnj/CVsV3d^iI4=FO9bRONOb`N/+1y,GK-O~N?~Ok&3[BYG^~OX~N+z4f~OW~NL~OTt!_U2?wK`~NLg/~NNA?i;~Mx~NG~NN:EN7~OR~Mn~NA!+/W4P5fL;;;k\'@TF?Dd(~O[)~O^B2l~NB~Oc9~OTSKT~OZw~NG~P\"~OQ5Bcn~OQcz~Of(.GNN29~OqC7&>955~P\"M8\"VDI~Ou~O_=.G\\]KyVFET]Qq~NG~O^B5K<d29o8W8L(U~Lhh?XI_s8NKFNp[OYBH+NUm<ETNxIT>_J~OPLNc8GQB~Q9~MThONc~OZB~L&~P80n:S~MmB~Po~OV%BTE~N6m~OSgF7I;9:H=-9iYEH]4(PX^;MlG~OQaa5ELWOXx+BG79~MK`DBH]R~Lc5HCM]e~Om!g~OPTj~OZ~P/~P_^@enda~O[~PgmfJ{~ObV~O[~P+UA~NJ?QQ;^,,~N:DB9~NIf4),\"(3~N(~M{)J/=%#]~N#W5UKZD6\'*~Odo|3c~M_3;B_yQO~P7~P#DSBngE*HaA(eVeV`T6>RYoU~OW,~NDH,$335&B6~NI~NM5~NG~NO@1B`W~NOM3eCPNVS0TVB[b[XNd@LWn]NS/GRRMO<K\'BU<HABMGfJ6>)2-WO~NJ(~N5~N)~N)bj~P&I[~N#~MD~M}~Mb~OT]HY@RQ4l2<ZP[HM7BPQclOO@?RNBHJJ6CH5DUQKFj)IYmY^M~NE1(~OVz~Ow~P&ZB)~MiI<~PT~N1~Ob\"~OcR~Mw~Ov~N8~P)~M}~N\'~OVR~OlsF7~ORy\\Y#O^ClVuS[Vl}lC@ADO~NKp/nyi~O_1~OP/Jc9MjT<k~Ow~N%y.5]pN~NB~N7DU~NC01F~NBD?HleD~N(DD1HOw0,=OKCFP~O]bPYuRJNCZC^Zc~OP4=/`;\'8A/@\'cr3i)=YXBBHOCW@V?RW_>94JELQ[~NK~N!~NI-FMH~NE~Mn~M}~M|~N>H>@~N)~Mp~Mk~N+~N1tJL~N:~N$~N\'~NF~N@~Q|UI1~N/~N)~NJ~T!~T>F213g~Qz~TW~T5UUEKK?TlW^X:C?HIKWV~ND~N,~NC:rN<~N>~Md~M|~N(#d:G~N(~Mt~Mp~My~NAyPD~N<~N(~Mo~MM~Mz~P|JB@~NJ~N/~Mo~P5~Ssf&]`z~Q)~Sj~T!LXeJp~Og+gA;@>Rj~Q,~Qx;I<&~NJ?~Px~Q,U]A~NF&>O~OuOK?~ND,)CsIjV*~NOC<VpNB3=)7f~NFT<:fa[KQF\\2~OR~P0U~P!:;K+~NOa~Q7~Q]8C@~NO2+~M-~P[BL3~NF~NO~N=#~P,OMA)~NC~NC2}\\WO~ND~NOG8Go\\367$@E0LCGXQ6U~PLy>dNDc~Pxd~OV-IK?ou!G>D<B-;7;aYPNMAVg?9EeXf=;NE``H=eG<??JG~NK~PHJ,?RH5x~Po~NK~O{~OQ8~Ov~N7~PO$~N1\"8Ki*RFy6f~N?oJ?gKMUUKDV/T/_ccV%~O_[mIPM[l3~N=dH8_JA~O[E8DJENy~P/KNK42~NH%HBI@~NBXvFON~NB~Mb\'GZ]GT~N%Q^~ORyt9E<oc~ORY>Rc]q_M8\"N*bbYb~NI~Ow~M=<~OrKAK~PM~O^j7JZ2~N8`=#&G:Gmv:LL~ND~N6#~NLUk%Z+~OVwheWn\"Bb^]h`~NOi>Y\\#z~N*~Oa~Oc\\H\\\\gQ~N/davG~N?~N2~M^vn~NA~P;~Ms~Ok~NN!~N85~P+k;XiZI~PHM~N4fPR~Om~O^l~NB~NJ+Q^~N7~N8G9~NO~N;AYx~P4RbcBbm*F~N@~NMt8YbI~NFTE~N9~OW@~OX)~Oq~On)~OV~OkL~MV~NDbk~Ob~Old>\\k~NJq~OZs;bbfw~OaR~NL~NGVQ;mrL~N7~NJdi&p?:T~OScZ)F\"~NG<_y`<~NJ=,~NKIctFXwn]f~Oe~NL!]~NNhd@+W~Mny`vf~NI7^pGn|G@=>o>X_O;A7]IXO-3~NAEG9UMF@;5~N+)8f.Sk@&~NC~OpE~OX~NLa~M^~P>jlg_kt~Mp~P4ML\\Y~NL~NA~N;~NL_dvFViL/?|AEVP8jWl&\\NS68AR>lL7~Mxn5`)DnTPd~PV~Oq~MN~N#`~NN8R~P,~Oo~N27$J~NAP.)/2DG?-~NB]JAN;+-)R5GI0R:~P\'~NFE:EJ?_~P)~Oz8@DH9E~P5JR~NOj:6D~QDi~M,~N/dGbK~R(~O{~N.32(2)~P9W;~NNV<:\'~P`ISLS(~NON~P#:ANV(9N~OSkF31JcPM~OZ>G?I0PDx,><=&CZUPIaOOkS]GDadDOgIQ`bZ^]cF~NDYSPUB~NI$:IIWYM~Oq~M[XQUZU[~Kj~LEQjY_U`~QH~M(~OZQwOQiaCVGX^Cb>bGIbSVJjBZQZ^Wcn<~NFOYD[;~N!\'HIHNTX~PC~P=S[QUUT~Kn~L>sVdSYK~QX~Lw~Ou=~OVDGo~QK~LJ~P2~Oc9m{@~MUpScUV9{gfd+Y/-&y{GJI]?I~NJWaSM6:HLC,Q?lQL~Oe\'?M<Lho~N9}9HGTa~N>~MW~Ol\"~NE`W~NG\\~Oe~PU+?DZ?5~N;*=1R1.Z~NEHG^Qb;K8X_WAJCFQW~N>OWT?I~N\"5@MT;%DJ~NL[RL]4~OP~NO[`IPUW`sQVQW7dX~OR~O[}XNP6D^~OT}IO~NLIDq}pEHAfx`n~O`cFWdkS~NL~OTV%~N\"aZZ^$~N6~M6~M6~L{}5amWHKCY?x@Rf2]n~O]yd89^%TSemQB\'q~N,~O_ZIHDPu~Oam~O^~OZKr~N<@~OQ~NM1]c~Od^~Oh~Ms;K&~NI~MW~M:~M>~PBPY^IY8dX~Mt~NA^M[i~NK:~N6|\\w~Oj\\KMcA^WVeg~NM~N;?~N2XdUOJ~N9HiE.;@{~MjkwQ2\\PO~OyM!Nf[5*~Mv~O]V[b)vE.~Oz*Glv~NKo1~Ox=xRf~Of~NOX~NI.RQDr\'N@VN>m~NJj~NNj~NL=X6Rf1~Oife>=m(~N1T9lA4]FAE~NN#<I~NL<ml[-C~NJ~NG~Myob;X[$.0~O]PhWRULYjPhMb_hX~Ow~OT<USg~OY~OitkeTSpe~Oa]xz}k~OcQH~N<?+G~NHE~N;~N>NBRe0:~N>~N#fnk\'F9^SS65;G(~NHE?*=H925J~OQ*?O<><?C~Oq3)>H~O^oUvZOTU~O^o\'n\\~N$[~Mr~OW~N3~Oh(9AD8$LV*Dj,%M~NA~O]SV;>7#s?d?EE^KiUV~OTKRf[a~O[AANSVX~OT~N2~OlJ~OXe`~P]~N4B!~NI~Mz~N7/~OPHza%~NO~N<O~NM~NO~O^P~NDQA~NM=\'AQ{ivCBEKdjog~OcbFSQCt~OrWlEV~N:-fxuO{~NDgv4a~Or~Ox~NG~NCNW]T^ayVgtQy_oocmML~OP~Oa^~ORsT8M>^~O]e~OS29;@/M}*~ND~N1:9LpZ~N;~N<~Mp\'Mjy|~Ou~NN~Mw\';F]j~N1+~OX~NJt8l?~O}~O^NZNO~Ov*o167uImt;P3HSV?r~OTGTNK-tw~N2~NMLwJF/Z~NM~Ml~M`~O]g%~OtA/~Mt~N(PB>~OYH~PP~NL|ADNb_&~N\"=<@f~Mx~N;e;NKMcR$mY9\\c8AZ[CCTQQDQ+:=AJOP,OUeOCD~OpK~PLd6P9KT~P^~N2;R%Z/~Os~Lz?eU;~N>U<~OZ~OeahZabcFDP.jZG\"~Oei9PqemL~NDA[8H=?Ot~N1yMHNh~OV~N1>{[t?^~Mk~OY~Mt1LTldW~PL~N,~M|iPUJGZ~NHRUr#MSATfb<>VXXAK{G8GFSMIVfbIHT-~OVCGQTE7~Q/].L@3It~KZ~Q\'4aAR~N--~P6~N=LTPW^hJ;We$FOG}Oa9CYZY:Lc0FWG9_Z`tRG_]~MwuUOTSu~NA~P!~MN]MRM~OW1~Rq~S*~Qk~P)b^Ej~Ry~PA~Mr~N.#/\\B~Qc~M]~MO~My~M{~N+FN~O`~N1~M|~Mp~Mz~N0>FJ.~N#~My~Mn~NJQJHC~N8~N.~NIb`I6L;BFaWOjWCBO8H=~S:~T,~Rq~Ou@H~NA.~Sf~RrE~N2~N9\"d[~RL~NH~NE~N0~Mq~N\'FC~Of~N2~N*~Mg~Ms~N6HL\"~NF~Mv~Mt~Mu~NEYU,)~NE~N%~NH\\f4~NINPJPfQHDTA=MCM8~NJ~NOLRAXRW,?AWYOLT~L^~Mjp~Oa\\NQY~K<~Qb~P%~Ob~OkYII~KC{~Oy~O_~Ol?QU~N&7g~OdTODC]iMFgPTX1\\QQL]IGKAPOXPKT9GHeUNSS~LV~Men~O[dOTT~J{~N>~On~Oc~OjUQS~K3~O^~Ow~O`~OjESP~N,Os~Oo]MIJjUPNjSULTbOTQd\\I~MN]8M39s~M2~ORC[Ye`~OTz4bTkXXc:2lZsvZWOKf[~OPvh_/:oQJW`h.dlp[hane~LbK?P5Da~M!O~MihR>p~Mq~PK~MW~M.=\"~NLD~M)~N2p0UVMZ<aG!Q|n6%FT(L{{M\'BqDJ1R\\7g~Mk~M=?~NH*=~M0~MqC~MclPUn~MKw\\R\\MHYAj\\<e<B=#i@U>XBX?ZRXU[R4Ia<`gQ[#A,??Wnft~MQ~MR]<AY^u~Pc~Kx#~OR7HU~O_piBDMQT]B1`P\\>LL!6FJEQ>LCRJ@QUV5NZA`^UM9l~N+NSQVmJ~Pt~N3=[0x~O]S~Pe~KU\"xI\"\\r~M`~Pi~Ofr~OU~OVj+c~O]fqF4Je&~Oa[S;7*~OYb~N=g=05A73u_dH4**8~OPFAUZ8G8{^[>:2UEUbbKyJ~OlYm~Oa}^AW~My~O}~Q(pefm8~N%~N=~OeW~ORQK=~N,0IVhH3#FW9>@**9?nLG6,(BE~OPn0}~NH.YsOo~O\\UH^}uH[p+)~MN~M/-~Mi~P=~N<0~N%Ui~M{~NL~NHz^[acWdE~OUWZM?S]~OW!ttFBq~OW~NG~O[l[b\\xO}A~OPuzmlqWtL^M~OS~O[~NC~OP&N$~MS~M$~N!~NAoGF~M|~Mx~M]~M5^X:EB.~NN>,Q~OTcLVRYgweodMJ[~OQYa~O\\d|ryjI~Oh>yz~OP_NnyfbfhV_W&IP<U7`3)`SBY`~NLLblFY~OQSPLTjHC~OSw5QRQYAI]7H|^PQ4FaX~N=@_IUQU`v~NJ\\[4CQ~NK)e\\\'tp~NH~Pq*RM@GX?q^\\R!~NJ5>r\\PSL~N;~MlHhTN\\H?%`=<LWY6aUZHtOKDEs~OUL*RNcm~N2i~Oz:Z<T_5y#bRPQOX~NJQdDNPNUNRC6~N9~N5;MO=S8~N<~N72UBo?J~NB~NF:Bs~NM~N&1~N\'~M{=2!jt~Q\'^~P/~Q/*hELLRAXH7W_LHGSSV^OaNRcDIFOL~NL~N>GRWi7$~N3~N-%BX)5.*~NO2D)VD~Om~O_~Of~ORbsO~Odl~Q*~Pq~OW~O\\~N4n9Ih`EKlTLPX[^F[/STZWK_=RE=ISSVF\\LQCKGQF*~ND%~N+J4<Qm~OS~N%~N!-I~NJ8~P+~N/+~MK~N\'9,9r>7aN\\SM:AKMOV?]:M`a`OdB5CFLEBVF]jfN[QH=?6JD`25Io~P\'~P$XcG~NK4F~Mc~PX:~OU~NO.~N&cP5XCdOGNh7VNYFBB4<AYJLsK5JR>l_OJ@mKKPg`~N/Yz{J@.A~P@~Mc~OZY7=~OVQ~NJ~PKa(UP*~OPUJD5U\\5*YXO`TaTb]\\PAe]]SM11HKrXWD~NK06FY\\Q~N<~MB~Me(5GRCV~MN~Mb,5Or]~O[fK7L^hl`\'[<MP@M~MD\\NJTRNQ~NBm2K]S[DOTeUYVVO~O[MxUTX\\`~OTxazGQV@\'~O]=S^O?]~MvWR0RS@+0BNR?@a~Onx5(<B88?B~N60IfLPP~Ms~N.}fTVUW~ME~N0~OXf[QS]~MJ~MvNuHHOL~M]~N4~N8@TAE9eHTECKTtKOx~On~OZc~P3~Oid`p~OP`d~O^iK5PiUcQYN??ZbGLj5\'5MNePm0I7S-~NNN~O]^*4!~NC~Mq~Mt~OW~NCo~NHJ&~Mi~N\'Vj~O[~Odu~P2~MK~PF~PJPWtLdA~OUdPGTSv#~OetL7:YS;T)~NB9/OQAEc~OVMXXOIxEH!A~N=&.~MV~Om9~OQ~NI,\'~Mw~N*tF$~ND~O^+<p;7S~M\"~OkkI~Oy~NIJ~N*?xu\'gw~ORRUIcK~OQr~N>eUM:NqDOA5bGIUq\\{1V_R{M*9X4Yd.[a~NG~Mb]nuFc02~Ll~NDi~O\\nN@/~N.~MOPaP>t0~NE8(@~OTrdPW;Ygk\\?\\UE\\f~OZlGEBMXl~OUnA~ORMOMJS!J\\\'~O]5~Ri~JY~MDTn]B)~Q;~Ly~PoQnX7jUO~MNE>ALJ~OSU~OijR~OU@P@DJEd~NLi7E:TTS\\X=YTKQQ]7POV[meEg~Oi~Ps~J|~MQ\\fgMH~Mf~M=~P#5w[E\\$q~N;VP=MK~O^XmMR|NL:DWVb+^=3E6OQKW<Z^2QGV=UJeW~O]=AQ\\R^_DqNA3kI?IVEU<ISZSHT~Os~OaQ~N@OkAA~Oh~O}v~NL0MdHVbY~M^~NISNXM_~M^~MJ~Q6T~NJbLW(~Q3~LDzGOMpU-^F{GC>e?IXDGJ7FKzT@C~Og~Oe>%^fH=~Op~OQ@~N.~N7QVLT^~Mq~MK(29V[7~My~L3~OseSvi4,~Ok~Lh~O[`Xae&.3;~NOQOAab4MIP;X:LT[UTFIIHxKHLRXE<R5X7Sg=1~NLR~OZ6CXZ~OW~OlB:)`~OQ~M<~Pi~QEb)e5i1E~Om?DI]Rc~NIKVRP$^>^]?ZK\\I`8$XRT::?LrHZ<yEm<^M9(UJDC~N@~P-Liw~NE~Ob^~Ozs?of`~OV~Ow7<R#/T~OQrlS~N/~NLX~Oo3c@YDV~Oa;IE,]YlMAMO#g|CEDYLG~OdMVPJH)],>4,D1dJ~OR~NCv#X[<C~NCj9E^~Mn~OZxg~NAQ~OQ~M|~OR%U\"O~OT~MzC;MOON~MpE8s>Mo~N#r>p7pI>tJOFdQ0?F>^-D/~Od~N:hn~Mw~P#~M|~M,~PBaV~NH~NDJ~P/~P*!F@XW~OPqk~OgJD3QFqWBL`QPMAc-~NO[BMB^~NH*OXILEBjm{IMMNWf\':NR~P7-~M7~Lg~O}XBS>B~N\"?\\X^E<f~Oz~OYq8C=ZX~O[bK3]lMH9H3XlG:]M0?OT7RKcdZ^cAHLbX\'IcBQMJZVWGPZTJFdLTUOWWZI>SHVSENRYONPH{L@/F1\\7F8gjI07Z~PC~OU%<~P!k~N.~PB~NB~Lx~OrHRFURNYM_IR^UR3LCQH_XHSX>ONTSZH;SHIONgZHJK.=_[~NEDB?H3H3pn~PR~N!W~Mm~NN~M!~Ow~MOP~N+!RgX<T\"~N)-~OXc`A1~M$~MpRuaJJ`~Lq.g@=DED~NA^HYYA6X~Ox~OTXeujRn/~ObnPae5=~Oa~OPpZWh<G~NF~Mm~NKW|TRF~Mw~My[r\\eG]~M%PqeZc<U~N<~P#rX`MVR,e]>D~NM>V~PGu:XdpM^~M|~O^I\\d_SE|x6~OQ6lMCSW~OQ~O]~Oa~OtNoV~O]wLIm~OdY~OT[6~N<~NEBb~OY~OXQ~NF~Mr~Mz~NFG~OW~O\\W%~Mr~Mf~NIK~O\\~OtpQ\"~NIIZ~OXBUrB=U|Whb~Oe~Oc~OR~OfYnO~Oc~OZ~OV~Oe~OQuZ~OTtvXZu~Okz~ObhM~NJ~N?Glu~OZK~NO~M^~Mc#X~OZ~Op]~ND~Md~Ml#Z~O^~O^vQ~N@~NGOu~Oan~OSsTP|h~OVVq~Oc~Oi~O]~P#nY~NB~OUk(^~N=4~M{G^6<QUz~OWRd_3?OSQ>UYvn@}:_X$RqJ`OS.3Juh1@~N7t.y]z~N=~Op~Ot~M\\~OgG(~Oi~ND~Mw~Oj\'04,1Lh\'B57KC~NL~ND(OY~NNKdQKRwvV@n_=Ho.mGiT^(q2VZF7O3p|fX(~NI~NB~MS~OpTL4~NJ!x~N\'~NAR.@~N1~NC~P$+a,,$~NLG~NKcNF#(h+Z^3D[fAhA4K^KRmWJoCF~P!gN{s~Oh~Mxw~OU~OZpO~ORo{x;Q~OjN~MwW7~O]~M\\~P-~M&~NA~Ou~NKb\"\'S:a~N+npSK10hi~OW\">N?OJ~OT4;~NOaH+OH^VbFGT9&~Oo~NE~OQ~OZ^h~Oj~P%~N)~Oq5sf~P/~NM~N8~P?~N>~P*~N*8~N!6_~O^~O|8~N#~NF~Mr~N$43NQ~NI&~N3(q]wuA51?`J`~OlL7GGM<lkw@KH?_JaQI85LOGPZIJP;~PX~M6~PM~N9~Oj~N4Y~NHd~Oo~OwZj~NG~Mbo7OROgB~NHuUnlPBFIAEZ~Of9HFT9;\\cv@]MFGU_KN30ObBKYIGZS~Mf|~Mo~M7~MA~MD~LM~LvT~NBDZ6h~NMART1~N..NVCQR~NGwo\\bSWEZa^JKSIWSiGMBD=_BODDj2LAHYSE?H~L2{~Ox~OY~OSz~PJ~PF@~MP1(--*~NJdb\"~N\"$O[gJL~N=Qog`M>CVg^WZ=MPWmNPUT``TWHU`VB<QM@JJG*L82rZ?W$cDAUROH3TECK?UL~NAK79@<I0h~NKB?5<?T~Pz{~NG21ESj~L\'~P4~N&FHEVB~R$~Pc~M[~OmakVM~OW:8>Y\\rH~OqSTGQdZ3`R@C>IC<E>8H.=J:,D\";,?I\\x~NNW7:HE_~KqKV6DLha~QP~On~Ob|~OT62G~OR^N9QCO5s@x`D],GTNlhXWP-?QCaliVSTM@HbVOTQ]D~NA~N>NN>0~O\\1>A\\~M^%~P)~N,)^:FV~M;~N$3nNlT~OY~MZn~P*gd;M~P\"cSOQY^FFI?RAN`pO=R>[kWc_Lvd~NO~NF~N@VET~N6b~O_NBA~PEP~O`~M66EAm~Pk~L}\'-UWFAw~OqSoKAB~NK\"~Q5OET:7?~O]~Oy9H3\'~NL6s~On@dM(~NH~NH~OZ~P#:QeL)%r~PLY=@N@[~Ob]V:5BGV~O[~LZ|~NErL\'>K\"FbMAMX]VBM74)Sk~Oz-OU#~NG~N.~N8~OZMQP~N@~ND~NN~N;~OP[n\\;$Nk~P\'2OKOD-~OoWk4GZ5b~OYx~MiRans~Oh~M9?v~O`Q[W~O^~P$T.@AKEYLJCUDKC][;9=COFaSAFUSiqSQC74A^cY3PyJKKNIP_~Q8~M5~N?~M(~M^~Mu~PO~Q5f+/I3~N%dv\\#7@8O67Q>:L?RQL?7>GGgXA=>Gi~O[T[DBGQTZe7Sx2:^`C1t~LN~O^~NOG=RZ~Md~P*~M|=PGI~OSqT><W=PASdTRgdFJXJSGuaMT;cZQPPLO3~MH~OtbJYIL@~Pz~MNMF[N~Oy~NO</SABmRR~MZ~MaADGE~MkT*^3J?IM[:EMkb?@DSEKnc=FJK73ERINL}~P<@6HP~N?&~QZ~P57=MB~N9^~MPZI\\B1cB~Ov5@A<Q$DUhVHmxtgG^`Z\\E>AWNO}pP`K\'N5~N#~N=~NN~NH~N64]bUgY~N,~Og~N2~P[K~OX5z~L?~P&+EVUXP~N6R*~N@GPD<#.>eYRopWEK`oabdl`VVCMUAQA>X\"~N)~Ml~N$G5~OQ4}J?~NO~O^L~Ouyf~O[gB~MiqUD?GXg<BjZ9FLe3~LzY:IKER~OP~M2ZTD8+)^~M=TPD%4Gf~LuH;;49U{~M\\_]@OOM~N5~L@b>H@_UH~LeeINR_g#~OPAp<FRd~N$~PVQQIK;Y~NA~OcTTD:50I~PQVJF136Q~PZNKC?>G~NM~Pf,pCYWl~N)~OZMkG>QhB2~Os{(@P~N7DC~O`4JIH$-.#5Z(TM7k8n=RNE,3~MpVoDT=*,~NMVDb\'C1~NM~NB~O\\!d85VHW~Oq~Mn=EY~NN8~M\"J~N=(%*W~OW~PH]]E/7,H~O}jb$@?:$wfL8U2&D~OUr~ORUT085@bVpO8$J~QQu8RF>p3~Q3~P0~ND\'B~NN#A~Lb~O^2L`d~N8~OS~P*RV8RTJCJ>C/47Q=3JjPR;XY~OXSqDB71F~MF~O]ibgfKG~OpA~OPD4JR^~M1GGtI~Ob~NNg~PyNC:D?_~N6~Mb5WIQJ=H\'?,809E4WLjEZ.PD\"~OP~On`L:,Y~Mzb~O_~OPU]P?~MqE^MGHNZ~Q+8/MalM=d~Mr~Od~N!~Os~Oa(~OU~OT~OXNa=d~OYN~NN~N95\'mN<[~N>>>g=~OX`y5#-oY5~Oa,:~NK~NN~NOZS~OTlo(Y7nQhHM<=R<js@~NHqLfES;\\~N@VmX_Ute3~NGB9X~OUd~OS*~N.~NJ.N|~OSI[\"\'$+{~OYRN-~N64]~OXuDF~N8Q~NKY~O`~O[%ki<-!a~OZ~NGB]UVPRZ~MCVLU<Sd~O\\bQTC5Md`LTB)-~PGJNTJIP~PI)!>ZNotE$8URQ~OSd6:VF\\~M\'OUZXVPBz8IeI7~Op~K9EHXAXw~M!~OrNZ86W~LUo;_A98~LnTHLGMX~Lt+/@\\@u~LSQ05U;~O^~L|i@CUEE~K;~O`G[aB8t~O^~O{PYc~NF~P#C=~OR\'adp~M|~N0~Odo6Y?~O`V~NGIG8MF]{!(T7)H~OdI=KLnP7$5DP_~N!Vm5~Os6Z,I~NBZ~P$~N>~O}~OT~OQ~OkToUi~M|yl}gpJ~N\'Pfk\\y~OS~NN~NO~M{NcOQKF~N:~MmJTPt[)\"~NCSJ29[9$~NGER~N08:f~OyPja<1A1qr~Mr~M]~Mp~M`~M|2~ME9}~PK\"#(-~OsD~N/63~NB\'(SE579Y\\/:RP18adB-IBCG7CCG&5^>+?NK9^V=;bQP]~R+~Qb~N>~P.~OXY~P_~Q.~P*~P-2~NA~N54[a~ND<~NE-%:V:%43~O[{0:-\\BFje<9KP>M5EGGP6SU51ZDJMIIFHJS}!A]\\OB@a~O|)TCKbR5~O]~NKkFbSU[;EL^W[OM~N.75MSKFQ|~PlRuInFO~KA*~NIYVh2Y~OgrCn?g\'7];SaZNH=~NO0bJLUDd;;T;_W@Hf)JKKNSP~Oe0~NJTNJL\\G~N(Yw[bKP~M4~OXBYdJ~NAH~N;^QW_fj~OaKGLQEIKXH;UKWGILBDUXZNHFGK=~Ma~M^BQAK<~NG~Mg~Mm~N8JQUa=~M}~M{4QWGV~Ou~O{~O}~P+WHf~OQ~P}~S$~S.~Pr_iH>VMJPA>NAPOKPQ_BAOOWJHCND@~M`~Mc:>I[J~N>~Mh~Me~N1LR^`7~Mb~M]8]\\Tf~P9~M-~M>~P0eEn@~Ps~Rm~R9~PbCz~P`~NI?LIJ~N@y+~ORDDHS<,~NL~N2A<`FSW~N>^W`VVKR~NAumZgHB]%a\\kQPOI~M&~MShAMY>)~Os~N5YNZh~NDr*Jgc=M_\'~M9~P0@I^`wEF@Q@^EXhJ9OfhgDQFXc^ZFO^=5QiDQ_E~Q?]:_WT\"W~J\\~Pd.JLS(3M[?P`99hR1[dRF/RL[<~OV@ZWITJec_eYMPO~NGLk~OYWVFW~NG~Ms~NA~O^~OTm7=2~N47~N;vZm;5G~P1{~MJ1~NDlJf9B>?`ESdRSJ]XaIq;YTSV8oe^UX~NJUV\"|\\~ORYn4A\'~N,16H~NMS.&J~Mv~PNY~OT~NB|I3O~PG~Ka~P*b\\+^.6\\MQGOJ-~NM9;^Pl5z+m~NM0T$\\J^fWQ@|KZAk~N?~Mu79o.U~NMo[7d]{~N:~Q!~P;n~N0W~NLB%~Q5~M$mB-I<PFMNZEF05MN<L[KV2\\5<^KOUpXL1<rHASqQ~NJ+G_]_Z~OP:O:HXf~NM~OPJa{D-h~On~MUQNGV?bB~PN;YB@<\\~NC~P:GI9/=*(~P>ZA\\_f.~NF~PnM;Ubj=#~Q$IV=H=/~N6~P!:h@:AT~N/~P\"[I;Y>z-~P5J8CTOZC~LNac=8<P0~KkM<?.J6h~M[[LPMX~NL.~MSPCNZU%*~MBTH2E<4?~MU^YJ7GT(~L2[?;NO]6~L`~PH~N#~ND:Nu~O{~M|~Moqo~NNJ/~N?~Oq[xJT\\[`1X~NNF?C&~Oa~ORC@jIO~OR~OY~NJs@b+wMwVR0UaZm;~N20k2~OT~NF\\~Mjy~P>a@tQ7~ND~N4~Ox~O]-)~NJ(~N(aWN\\+<@~OPA_/ED>`~O].31AB;~Oq~OSTK~NBRFU\\hb05[cD~NK%1VWT`YT\"m~Or~MO~M^~N3M?~OT5~Pl~P-~NKdY@~NI~NC~Q&~Oci<7GU+~Ox@QGF8DH~O[?6I98NMzH73;>]|g}?JBD~NNe`~O^YETCC5~O|~Mb~O\\~N,~NAL~Mu@~PN~PF{63T&-4~N45^)?T0~OqPHD<2J@>C<BV=KL~Or~NFB\"JEDa~OV~OsJIJE?[~OxPg.GZU4~P1U~O_B)~N:~OQ~OvFc661.~P,~OaCH-8+MG~M<A>8~NO$1W~NNKL>(+:9ZBNX67;7lN.DXK4=DQ`OX1@dM7;k^~Of~Pd~RF~R%isV16B~PW~PZ7?/.6!~NI~L_GE8$~NM9VNAT28,BcaURZ5+@EU8-QW7HQ>aUZ@;^DE~MS~PCA{~N4F~N@~N/l~O}~NO2]f~Oc~OVG<~O^Aq\\\\b`*BA`&L1MckbmRH`RHI1]FQ;b~Oi~OR\\BD~OUp~N/~N3f#~OT<~Oh~N)~N<NJE`U~MrVU~MrVHR4!ZaKnA\\Lp[CNDfG>U7MIVKO7SV=UUFPBENaBWXCO~MtF~OWW[XQ@<~N)A~M[~M{8o++~OV~N@~OTZV@m^M~N=,~P0[~NLSZ-ZW\\FD1ARNDQ=e^0;!Y[dq;PS/4d{~OPTI^uH>GYF^h~Od~M(H~Og~N0[E~NO;~Pe~Q+T~NL~NNvD,D~N>B7TFDLcnZP+N3<P8^LTDC&lnDH91A<)?}tKTL~NE[e/t9Po1VB3Ij~Ob~N(381ZI^^~NGKUP:PX*~Oc:N~N3XdB[dLCO@etra}a7o~OYLK0t68P\\~NMK~Pq<~N9QXT(~P3~QF~NAU4dU@`U8E~NNZ:d\\\\`[e?<B*fk?~N+TMT)HUALgO~NI~N9@~NMY~N;v~N?th~N7L9-6~NN~Om~MzB~NG:g6N~MlD~Ms;{\\t^qp~OQR~P\";4(J~OZjbA*~NL~NE)Hg_/~NO$#~NG*Y^~ND~NC~N@~NE~N?/]-!~N;~NE~N?~NG(5x~P/&&+/~Oe~OS]pcb^p\\~OS~Oq~OTXXTkh~Or\\~OlyLOV~O\\cV_]SWa`tW_NGJYOZ@]WbTdMiRVd?O`XsF~OUU_Y[~OSL~OrMj__rE~O_~N=~NO~NF~N3~N3~MG~Q0~R5~NO/~NG6\"+~Qh~R6RN~NJ(!]a~QlXK1%&11~P[UW?~NC~NK+/~O]f~OPS>$.\\~OWZYc[SPvGn4GXP\'})~N-~N4J(H~MZ~PF~Q*~N<4FA9*~P=~Qr@R9~NO~NM~N\'~N?|g<~NM$~N3~NA\"~PKY?9~N<~NB-=qonR5!?Q~P@=Y^TSOde?OKIT%Ip~N?~O[~Ok~OQ8N?~OT~N//mweVfz1~Mz&ae{rY=J4:LVtTRY\\FQ8HhyqbDtO>k~OXdCJCnQ~N/D\"!~NH\'m~Ofs~NC~OZ~OuNd86u~NG;#,~OSu~O\\;x~NCX\"8M~OVE~N-QlC3EZ~O\\vP6%)a@~N**McEKW:~P,~OXoT\\X)~Oa~NDz~N7~OS)~N@~NM~O[M~Po(t?eBNd~PV~N6BXS?^O~O]DF@EQKS~Pt>@.>3K>~PnA78DQRDj&E/YaYO~OU~NFHOPLL3~Ok~NG[.RXQw~L>~NJfXDK54~Lg~NHBRQ8de~Mal@1=G>4~MSr,@5AYW~May;J;GDO~N&@:,VTGV~M$G@iLQIX~MX/<Qb=<G1X4A^^P\'~OP\\QYSIkhE>3`XF:<S+;P[UH4>~N@EWJ\\%cq=O>@c|i~NH~N\'\"!~N3Y#3b~O`~Pl~Lw~Pf7~Oz\\NLBXPO\\1QOIFISOgZF~NNldLBHG/AMTVBBS~NK<WLsIjj5,+Od]c~NA7-?~NMIY#~M`~M0~Ov~Q]~P@w~MX~Mc~M+~O^~Ltl:~P%$~O[?~K(~P.VLz~Oa~O]~N*x^4Ub`:iCLG\\mk{nBPZGK^.mgOgIULJX`[]\\Zz\\ZlDe*iON~M{n~Mw~NJ>_7~OT9~L*~MM~N?Ea~Oc~OR~NL~N%1`5qeN*;[LNfSPRSRbNLi?WVMsH_[]SJgSWi\\MGe`GHnD;~OqzQkXI9-C|D[SVyHPGBTYMS>Y@\\HPwz~ORKK>GFbmhYO,<[^:CPg+>)~NBUC~OqMGawX)|32{V[Q[/URkUe4~NK\'FicTEC(7MVgI<VZ_WKYX@_HIZU/Erlm~O_~N0~NObTqy#J5~NFGJ~NE]%ML~NA/FtLt#-@~OV>Y.mZ5K~NIQ\\$~OPUKSa{~Oc+~O\\~NG4Fd~Og~OnZ~M}]^FG+P(~NA~N6J{~ORg~N@KH~On~NNZ9PNCS(~O[bcSDTT~NFUWWB>N7-ANR;@=Y?N>2~OXl\'M~MsQESzq#K~N<PGXD~NM~NGl~O`CQ]DAn2~NG~O_MW~M}-u~NMO~M%0O\\MIOV~L$(iBFG\\n~M+mCA%L:L~La>(LRNVT~L^H*@C<QH~MLVRbgCMG~L5-AGA5ng~L*9O[UL?A~Oo0oYOK@S~P4~N6ZKMQ].~P\"~N\';P,NWK~Pn~NE~NIZHSN`~P9~NH8TQ@@U~P)&Fl_D^F~O|~N@JGT9m3yGj?VCXQn~NOV~NM~N,~N7~Mep~OT~OVv\\K>~N!hVsMRMEVDAKJaK`]BPDPdIcv@ijVNOkcd_3MZ~NH~NM%pO.~Oa*~Ms~NH~N0~O^B3XZ!~N?>~N!N~P#VW~OlV\"~OesZ/KRPZ~NMFHMZvHjfJ>;WcQpR`qQ][DNlF~OkH*~Ow~OTST~N,g~OT4(~MU~NK@~MY[q\\O7Hk~P#~NBX*4RN)~P9141>0Z?~Ol0Kjg_>Z~P39<liK=P~Pd~M}~NNHC6P@~P\'~NF\\4<Gc@~Mmj~OQR^A<E~MOa[i=MP^~M\\Jk%&PMy~MmhM6D/UB~Mh~N5:OPW8J~Me~N4/DZ9@T~Mbb5H?5_E~Lo9i*:9ah~LzbX_<Ob?~N2KSBG%2~Od)~OXB(DD~N.<%B>C]4!JgT_hYI5uRo~O\\~ORHiYy^~Oj~Oxy^CeAc~Oli]ZT~N4JCc-]R]L&~MP%~OQm~N-IR\"v~OsE7Ga4`=&5GDIC;a5.IY^067J><b>_Jqs5Y=dzu>~ORxV?`=fM(4~OdMxd~M]GeBOU_[F~ORZT][OLI~N+Z{~OXSKIP~K\"{~P\"~O\\~O]DMF~Jy8~P!~OX~OhTHP~LY~MXi~OQg\\TXVCR_UOXR)Y?FPHRV2gHJPZMKqhZObJOD~N!HnwRJHH~K;~OS~Oy~Oi~O]MGN~KK~Qy~P#~O]~OeTKK~LP~M[r~OQkPVRY@HV]L\\D~N1+KLSQOLQS~OPS;Kypc$JUWLeBW[BT0fTCDLWWXpO<@a-YB~N5IHP_K<~N#~NLkE~O\\g[21E~OcKdm_;CW6~OaZ~N:<U7B~O^~NO75Z1SQs)[[:W~NL__MNHb{hh^`<S#}~O[&\'xPODQ~NLA~NLAcsgH>+N~Og~NH~OVMKDA~NL~PI~PR~Ot\\X%H7@~QC~Q)~OV0-D4g~Oc\'1aVNT#bAj8E~NF@LoEHTXVO@9/S3GOUa~P3~NCD73BDl~N+QF~N;%CI3=~P.lQ/*~NAS~OW~PK}I.+OP~Pj~P#ZE@7.7~O}ZonW>GV~Odj]^J`VN~ORM~N64C1==~OX~N3~N<~NF\'O=A~My-=~N1~NIFeKU[D<Q@VC+TI?:JSOF+~NO~NA0Ra46%~N6~N-~N74HOr~N<~N:~NF~N9~N@QB~LQ~N9%~NB~N>~NEBE~S.~S@M~N;~NA0Sc~T2~T:~RdqA>&O\\XDDM8^].5O?<GeSI5~NG~NE2`aAG\'~N@~N9~N+,TEb~N@~N5~N6~N@~NBEN~La~N0~Mx~N4~N>~NBOF~RO~Oq~N=~ND,0MN~Sz~S6~QQ~OQXfPoRu0Bm~M\\~P?~MhUOHXI~NN~PE~M-ciRRnZ~P*c.5,WMY1~PQv>4DXn<~N0_A?>K=RnMQ@uJ+~P#~N3o[@4r8~Ot5>)SM@~NMb0Z3Y@F\\&~Mmi?K_k~O}d~MOFK:OKM~Oc~N:XN1YM>ccU;)PEJM~NLV8FCCq~OP1QDAF>F`~O}~O]R>MQIZ~Pb=1SND6+~NG@VTqNK~N=\\e`lVaP~N2~MR\\JFO^~OU.~M@DN81gF~M|~Lq7.VI=~NG~Mr~Mb~OdfCHJUOsJ=;FOH0~Mw>=ZYWH~P1~MUNTXkKLl;SQiMOX`=TRGML~OQw{6ND:j^lm^+YHEAY\"PDWK:Gc~NJKd>LWPk5KfB7U_BXQX~ND~N4%bk[7A~N(QkoIE8A:d{=?TVo[e7j*~O\\yMTF0?~N#~M\\~OeV[kjt~MU~Og(k4[Ncj~NHhf3O+^rrL=~NJ~NH~NLWh\"<T~NB~OftOJzP6)vyD~NK~N9ZtKZD~OZh~Ot2WWH0a~Mi~Mty/BqH~Ox~Mp~MnENcZykck0QR||\\~Oi&a[~OXOhHe~Oi^AWHdfc~MYCQNf^e~Od~M[~NL\\Zg\\~O_2~LOg\\ajaY~L@,]l~OTTq~M9~N;/;UJUp~O`~OV\\K7aqm~OQ~P4~NMRel]tph~NIJcRl`SS~M)1HUYPi2~M$]Zohg<~MV~MNXEozY~Mz~MNkdzrTF~Ml&Qi]]1~Oe~N%@~QmQ>P[~Oc_~P&~O_CPXTU4/ATN\\~NL,wZsKUZ~NB*VY9MHq>WIO_v~N4[L\\PP4~N*~OTZQM2[V~N&~NHLw~NC~Mc~PM~N2PV`kTp`vUU\\kgOD\"QJ[LGTTGDUZ<)MVjH<SlS9JS\\2XoCAcK~N,YaXLiK~NEhxwYh?~N;Q|WX^kO`S~OSordGL/7wmXD61QU~Ov\\X>AE6O~PT~Ov^~NIJ~NN~M^~M`~PP~OZU.,~Mi~M[~N-]/;~N*~N0~N%nmjlh&ulOq~OR~OQ%iyk@M~OYUab]YG2XhT6Q]K<~Ow`SCMESXf3~NLKId~NK~Mv~P:~P0N6^F~L|~NG~Pb~N8~Ov~NKD~Mj,&~PJ~NOy~OWutU]vf~N/%S@~N;F1C43@kI@1P+Fm~OhN<QAK]cG@EQNScNOWK~NJ.WQRV6|~N?~NCPQ\\3Iz~Po~LHo4|~P-~P<~M3E~M^TP$2~Mz~P.AIBA3lg~N:2X~NOZN|YIbAGV_GHQ:J:eOKERf\\j[hN}B~N$biJf+P~NI8;d7YbI2iUG6]S~NCskS5LZo2~N=Vd<;Iw~N0>Yor/F~NL~N8)c#.WsI~N*\\AU)q~OP~P,F8<^h5Ke~OVN;KPc-hO>ae3KEfA|Z<;y0~OlJ~Mjc[<4~NOgG`Gt_>~NO~PO~N9~N>bDhL~N5g~NFlH:(~O[d~OXkV8^J;jhG~Oh~P*~OsM~Oh~O^a~Or`~Ot{O0=3/~N4V;aPUbu~OWP<1`\\?WD~N@-,`cFS.*/??3CQ0R2HBRQ?{z,M[-OY~Oz~NC/~Ojcd~Oz~OqN~O{~PBF~OPyd`~N:9V<U\\OSQW&#<?LL>\'A\'B?WP5~NO~NG8B6?4Q~NK4AOH7CzMukO46+;O;]VQB^9p~ORr?[_dFHfEW7KO\\~NM?aq|XDN1lt{uXJN~Md~NASONWY@~PH^~N8n<\\Ro~L2~PV~N0;?b\'~NJ~OX<HYFO3K<[fWSUbI`~O]QS?EAF~NHffmjEDZ~M7+dot\\MQ~N%H@RK[V[~Mm~LSGQbU\'G~M6H*~N#DoNqV<BO~Op~Le~Rx~T4SY?C9~N4~O[~SjKT~NI~N>~N9~N9~N+~Q6M,~NF~N0~N0~N;~N;$KC~NE~N-~N3~NG(P@RD~NM#~NK$Y_^bSABRZLWFNMFGsZBMG~PY~Ks~S7~T2\\TDG<~N0~SY~S{H[~NI~N=~ND~NM,~R<I7~NJ~N0~N5~N8~N.~N;LD)~NC~N%~N?~NLFIQK~NL~NB~ND(8[IoMC8Z3IPCSH;cK~Q9~Od[~Oaak.,~OZ~Q0RETJ~OWu+i~N:~NJBHQ:~O_B4~NK4,U0A`D1J<MC^IL3JPDf~NE~OfaYMJBFx~OSER?AQP~Of`~Os~OSlyL~N?~OW~OZpCE_WX7OW4J;ZP=/EbLD<O\\,LKDFH`I`;@I2=H+HTHC3*8ur_RXE=,f7\\IB9Ms3jDH>3eHgc~NHILW~NDuWTA\'J-Cc7F+J=Z~OUW-:Ah8~O\\~OU1$~N@5>_Z\\>)6;ZVbGFV~N9AI.;&:~Og~NO<^BbDp*~N>~N@;[_~N6Oy4uc1~OU:}]H(N?Ch>TNVsboq~OYr^CNOgL/~NG~P\"6`WVM+~OXRID^REnD?C^^H23o8WD;^VPSLK25Sm=DBGC9HCGI8h<)LOWsTk?]wQs~PO~NG~PUx~Ov~O`~N.~OcO5HK\\GG/MUNFCVXIFRFFBVNL`SB:4HX`_CG.+MUJJ#!=H-YA~Oc~P,~NN~NK(=~Op~O_~OuXJi~Oc~OUo~O\\~P\'~O[AH@tCEM~Ox~NABCZMO~Ot(<\"1DU[h;,~N>~NO:Gb_1~N:~NL):\\XS~N.~NE+.Pb5~P=~QB~NORHD~NO~N9~Om^D~NAd07~NI~O`~Oxh6NdR8~Oe~Oe:BDSPIaJ*62Dki~O`>~NK!~NH3dD~P2A&#~NKAEX~PW~O{?);I]~NA~R*~OkL\\J>~NC%~PpQ~N2;YHh~NI[Zo3UoP`43hvIoMCEf6MG]v~O]*TE^HWmB%XMVE\\c!1JDEH>7b~OQhqjR/~NGX~N)xWG0+,\"A0]qQ^y~NM&6~OZuV\\t}rq~OVZ_qJFj^2EYcSTK_MD_OL_~NF2HG~NKci,8~N,CQ4b~OZA\"$3Idk3~ND~NKPB+dUC~NAbY1TB+k~P)`Ig}[qkP6nYiUB7U@XuhpY[=L~NOR~Oe~N:0=\'NW~M`=Ups#5~N2E~Og~NJVx]A_SCWWK2Z<Q6?Z:saXRe]O~OyrNdgjl~O[U6X@Nx_~NGS*:7~NI~MR\"Y))6`~L|~N\'R~O]a\'z~N5~NFoyjSa~OSDPMHBTST~N@~NB#:~N>Sk~N2~N-\'~NH1*T;?~NI0NUSB_CHQONVcSdw~OVJ_w~OX\\Zx~OW~P,~O]o{~ORj~N@e~OUpA~N5~Oex~N5Zb~NO;bJb~N7~NM+?~NFJY~Mo~N/C51:]PAL0\\T7e~ND9jWgPa\\zF~NHADT>~NJ+~P?~O|~PF66~O}~P=~P\"t~M,~PBL~On[~MJo7~N!O@J,~NE~Myw9^VPY5NG_QA+I\\1iE,560;8~NM::-#21%~N3,YO(.4v~PFj%.~N6B~OR!~Py~Q(~NM{~O_f~Pp~P-O\"~NA~NG7G?.1HQKPV~NI-UG8<NFC;~ORA<=>\\E;fM.BC8821K-)D?R=~P=}^:Z^wl~Pm~P-~P\'~P3~P4~P,s~Ot~Ol;LFH~ORN~OU?CWGW(~N**aWFECh~Mq~QULG9&*~NH#~Lm<VR(DFP~OoZPJDIS+WD\\S[I\\&:n?SCDp4~O^/`MPV,~OzYVKI@8Z~Om6KRPN51~Ol~M]OOX,,G~Oj~PRW>d.)<m^=VfH9NG.7VGNE_gY:fI7G,b>~OW~Ly}~PC~Mj~Oc~NL.~Pw~P&~P9~NOEP~ObN~M=~Ml~M|19D8:~O_\\=NAIQk+m86AQeJH>]B?F%CYILY_>\\9c>C`FqI3[~Me1aqp~OTd~OQPNs^jwT~P@~P>k6GSbP.711>OPC>,(H:2K\\!-Hf31HS~NGmBeV?HZs\\[bVKUI~P-{<KP~OPH~NAk9)sqDbTH~NBn~Om8I_oLh~P$165C2Tf9-R@Z5~OP1h>@jJXKQpFLR_~NJ~N;@pA1_/p~N&]*YoKBJ]~MZ~OXB~N,E]pD~P5~OP~MsO.M3LF~M;YHA7gU~N26X<j@H=M/3gVYX~OWkk(ALhMAM8hFXU8+L+5S`WfBKU^HT~Oq~N2~OQSlWN]qj$ZadNa[~P\"f[Gri~NK+D@JgiP{~Os~O|^1~N7!8~M\\~MY~P?MN&~N>~N,~Myo=nG;9FOk~NIMN<RR_~OQNS\\fSVhfimpXZ]JooMJ^[GL~OY~PN=9I)0~NKW~P6iI~N;~N?~Mv~M:~Md~PF?<~NH~N@~N-~N.0w~OQREPE~O}~MR~QCVME@@h~LP~JXRbG?Ta~P/~R?PGX[Y+~NE5IN8L]~NN4~OSZJ?iQ\\WUGXRYSWr?KG[`DJ_d{IGWg~Oc~MJ~PrdNFA=:~L]~JdLOB?M:~L{~PkJIQY\\~NB5~O`FO=N_&E{GNEnQconJW_VLN\\U\\JX[HOhh~L>EPR\\M6}R~Mkd[FDgTzyaMFROHjX>UJMRNeS8UKBOHxq`GTSFSG~MU[QLYgM~M/.GMNQD~On~K[~PLGP2B~N\'%~OZ~PqaRRTo:3MJDGVYXR(UUPGLg^3EKOEM/C9D?OQJ]~Oc#eTT\\wD~M\"~OvA;JQ~NI^P`_Uc)#Z~LozFR26MM}~NNaM1I%?~Os_?NUPCSd]aGM5ML~O_TG^>=D9~Om-]_=%%Rd>Mj_Hj5~P\'~P5~Me~Oc&j_\\~Jn.~N3KN%)^~QB~PvTV/GA+h~NGMXYH3HBP^_Y9IFM5RUYPMU|i\\N^~NHBK*;`Z^9l/k~NC;r|~P4~P#~OjcG~N\'~NLZ~Ot~Ot?w4~NO1N_4(PbR@=%h0IT.5G\\^]XLO;i\';IHQaO2,~N&^L~OT~OdkR~NJZ=Z~N0P:~P(v~OV~Oc[3~NLLM0~P/~N/A$\\Y*-~OZ~OYOC8/:Z?j;?HL9/3\'~P&n<F~NKv^~OP~NE~P,~NO=Z?~N4~NBk~OScY[~N6~Od\'~PF9FOUR\"TF[<DLf2S.Q<93QCVUG>kuIQSMN<yvTPEQ[A?XmQf~NF~Mc+~NA-8~Lnj~MQqpB[`~Om~KZ~Ok5OUVLv~NB;VL@?bI:G?AP;PT~NHjGLzkEF[8I<y~OQV5~NFVZKhirYP~OQ9~NG\"~N>0~P*~Pi~Lnf\"<N(~P8~K]~M*~MN~ND~MLS~N4~O]~M0m~OVnUEXkw7J[x~OrGRC^hlfgs=W`lvdV]NSJZA`J>fV-aORMWb~NJUFPWZIBc~Q8`~M5~LS~Lh~MZ~M4~P6?JOk^[/A0WRx~OWYWKF_h^h~OQDIPitd]VV>?cIZ>JR<nWFUPNM<qPNPXV7V~NML~NM<~NK~Ml~N#kK~N;9~NO~NG~My~N!gU/~N2=(~NK~NA\'4=L5IC\'CV;=Z^8:S~OVee~Oa|Scftl~OR}iw~Oy~OPL~O]}~OS~OU~OV~OY!>g=+~NI~N3U=j~N5:EM~NH~N%PDP>D=BU~NED!Dti4+G8P[KDkOV~OyeQa]Ic~P*4~OdA]S~NK~PQ~NK~Oy~P7WA~OrA~P;~Or~NM}EBIL=XYtLJI@?Y~OWm&$,QMGx/~NN~NC~NH*MR~PE$!~NC~N=~NI68~Qj9\'\'(6FD~S>~P[~N=9G3MO~S;~Q&~L>>m\\2W~P3MDC]6DG~OsuDBLG8W~PK;!$(PNQP/~N@~NC~N3,LHT,~N=~NA~NC\"1=6~NJ~NH~NN%2L9~O\\~P9~N>=0K<H~Q/~PV~M$~N5nb&St><XMD~Os~Jx<RUCPo~LS~OqDWE8V~L5sLQ?<)~LWZRFHEY~L](<DZ:r~L(YEIQB~Og~LsyQKZSB~Je~O\\=OfK3~OX@YOTPRI~L|JN_BGj~OZQL`C\"OYieKH\'$~PEKJHCAI~PH)3Hc]gXL,ETOJ~OkjHF`JX~LpH[RZVWA=DNTSQ2oJ#B~M^~L_~Lq9US[_.~N#~N/QOIBJ~Oe~O_YLHREx~OS~OgZlQI`UPSLFQbSPNJN7V_YJTXEJR~N9{~NKt+O~N<OqY2~Ml~M*~M:7XOWRJ~Po~NC;NIAF~Ob~O`P>TMHm~OR~OeanHJ_VW[@I\\BgMK]RXW1=HcHPSI[UdV~NA$~O`~N\"&GF]CV~N0~Q9IMQ+fDJiFW0ZUgD~OS[KXZV4DA>FU`4Pkjg[NN]I(~L.F8PR5UU~PF~OdMVG>b&XM9S[_G~N6~N7[K<GjEeeIPAGQ[/~NKKOlKI$~NL/>NOc+6~NNoD7GYDG~P8~OQB)1X<~NH~OR~OUeHJRY\\<\\~NKaCL=P^D^VEiOLRCPYdjzLR[O_Rxx[YHb_Hc~OVYUVNXO9j0LO~K|v~MvO~Mc~L^~Pe~LGL5LJFUM\\RSI<AX\\]Q_>aEIbTM_fu~OPRYOMqW|sjZDQaak~O^\\O?1;HbX)~NF~N6~Pt~L_~N(~N\"~My~MF~M9~Q;~NC<YXg.21#/@*@=~NLK6N;8($!`E;8GHF0O)O5<?21<JX<B;TF~Lq~OeyqCJw~P/~PAW?~Om~PW~Q,~Of~R\"~QG~MtAY\\A].~ND8.AF&A~N3#82<7(LR-WF;B:39O)N.KJ0/~N.tF9AI:~NO~N-W~Ou(FJ~N;~PA~P3~P#~P,~O`I~NK~Mf~PL~PA)2^~NMcV-0EMML@DMpXV0(HHY0\'FEF?~NF&whW+GO):%~NLt~O]~N:E~N6~NBP~Oc~NM~P@~N4~N1~O_~Od~Ob~N?vt~O^~Oh~P<~M{t~N4\\B%A\\]WWZ[IQ;_*IC58C8K:V<P%*H~NFAdorD/~ND*94\\~OZo~NNY~OT~OaS~NHZ~OsuEH_~Oq~Ob8~P;~PZ~Ot~Mu~M{~Ob`c=X~O[~M^~Og~NH4Q4;)~N+~M0K^SANn1~OiY9NUJI:~OjK?GIH^~NHMIheK\\HHHfnZEKPK~OV~P&VU]\\:\\~Om~NIuO^~N?q~N,~MccQl~NA>~OT~N\'~NNJ_MRMs~OR~P0\\!{DOCJ(O8IEB^~NE~OPkUr=uB^~NN7ckI9@~OP:a~NGNJgU~Ow~N1~Op~N?J4~NLPck>~OWK;>;~NO0d.T~NL5G:In?V`L*8JI[@Q[2Se^]jmp_S^~N8>?p[_~P#~OP~OZ~ObTx~OU~NLx~Pq~Oo~Od~N=?6~N8;sJ`E>E>Tk(4A.GH5~NEc43+E\"C?r<T:4>>V9i]uk\\l~NO-~NMj~O`~OXgP<y~OdP^e\\^~PE~P)~LV!&fb~Or~N852T+-/)YoI%~Om~NIrZV~NAgpT{[I~NH~OX~OP6^I[<~OPDkd}YqrK`:-sJ]b3V~NO~N@HAiuXX~Mo~NHvagj<Fn~M_~N7~N3R~NOhx3~NJ~NM)RT]LqUMD-=^S~OW\\_dVTPJffRutF/vA~N5HR}C@fF@)QXx0~OUOG6X3qF~OQHKHD4E]p^9\"T(78Ya?3>=E<{oNG@2X_~Oe^qU$2\'Y~M2~P3~ObH8+?K~N3~P5~Oi:_~N.3~OP~NKc@JL@H,LX<NV0NV~O}fP+Z@.I~MO~NMLE]BY;~PG7bUF2CE~P\'~Oj![!7/X~LH~P:~N,WO(9S~PC~Q+~NB~OTi~NDMB<mZFS#s1~P/5^aXE5>N{a<>Xm[Iy~NF~NC~NIUH&~PI~NG~NC~N?~N>REL~QQ~Oo:~NC@QWF~N0~Om_M]BPby~OrM}~N9S=ZX2,WA0J1~OzAIS1PHi1~O]G6XNJO~Of~OaE~N?~N;SUP~Pb~P$1~NB~NOcM,~PS~Oh_53S:A[\\~Om1@N0Z~N$@k9B<1Xq\"h:<H~N:|G~Me~N@4&\'Tf[~NIMpiV)S+/~OfXZ^[1~NG4p[QsX;W~NGB[khFg)O%0Q5~N2e}XC>CKMg~N2DsF~NN7~MikB~O\\PJ~OX~Of{~N!U_brJ@z<f~ORqK>N\\d%xeA8\\T/F~P\"CLUZUY~M_XaSUagLx/)<NM/\'7H8OY37~Qq*VNA\\c-~LcJO=XQR$~N\'`G`sccW~N\'PAglhr^~N6ZRQ]cSV6A2GEB_B~LUHWRPEJ:~Q\\97Nj5,~O\\~KeUwGN_VT^S79PNFE~N=jPuZm{NMPC_dk}3y@Z?af8_-UQFTTP|F]?GQ>Hg~KWjHM]`F>~NDA,TeY<05=GU~Oz~Pj~P+<PVB-~NH~OW~POWb^ZN~NH~N@vXYZX6]\'U?T~NO#MK?F7=~My1XYNI?k~NMhQ[_Ir~MceGHUo9;iFCV~Og~P?~P&IbY;6~N:~M5~P5UkN^D~N3~N=lS:aP>d*[FlUCEVASTO~MlfaRRL5XtCYV]QPLcCpf:RU,]NWxsWUE~OXqgRjRMrTZSNKK~NL~NA&`:qlR~N*~N./2a__@~Lv~LxEKT\\`h~MQ~P{VK},iD\\~P\"0.f1O[~NFZ~OTjq1Y[qiWK@WVB0m[RNamcEA`]HJNK?~N1D+ymR~OX~LqZ~N,e)sYa~N6~MOGcOU?QgJibDQKCqJSa;4SQA>MjRG[rHsNFSEBlrA6[EUftH~NAs{k=K~Q(~N9~P$~N?~N1I^~NK~PP~L?~N\'VXREQ<VAc[ZB<NMFkl>5GGY>OO^ORJSX@BMZJv]L?_VJRgq~NO9HYUg~NI~ND~N5M~Odw~OUc~M<~J_~P#\'Q`PRHg~N+)0_/~NK+~P!52DW~NO~N/~NC2><XQV5~N=~N%`PDTFZ2\"w`nfH2HTy^uMJE$r~N6eVBMYW<,U2Cqk~Os~Om-6~NM.H6~OS~P3xA@/_vTzcN]QHap\\rsdbhPn>Ya~OTaeP/NGflrDGK:~NMw~OUcNZN2?xaeHWO8Nh@X\\GK)(dQgRPN;ALU`~OW~OQjEUF9E~ORi3~N/5^Pie;~Md~Mkj:%.1~Mz~Mj~MY~On?JOlDfd~PD~N(u8^EYVEoPZFGTQJ:XZAKPKhVGcyvl_gT6Do~OY~OS(BV^gX~O`U~M[;\\FIF+~Mc~O^~N;%bPi*+~N=~M5@~N6;B;4r~Q8+8QVI@~P0~O^WBGT10=~N@IJ:IMAIPDQIAOXKbHMDtgLT@j~MpGUJJ~N4~Mk~MsmgHEV::~M>~OneSJ{~MT~Q3~OZ9JJXP(~Mk1;\\S;@[v<G<NTZW~N=MK=UDa7~OmGVNq\\`>s~OV~OuLZUU~N7y~MCoKP1.~OP~LDdCPO\\C~Ox~N<Ql85]r~M)~P-7=JGirjWYZMCNcdeUJxBML<JMMSzUJ;\\FPQM`6mG?8dO\\LJSgVUdh2;qYPI/~NH~Lv~M@aMQD?~ND~MA~Lx~NCAWDET~N=~N(eAGzKLJ(yVVJzM>KsOFYRdGTj+UMVMEQrfRAH:IKX:BF[?A:.;LN)-1PGPN,~N9~N:+5c:J~NN~N5~N<!K~N\"85~N9~N=~N2~NI~N>~LuW~Od54%0~Of~Q\'~Pk~O[~PPD@~NM~Pj~QinHF?CPH^68PL<KE:RJQ7~NF3:P@J\'~N=~N:6J_5D~NL~N;~N:~NKE~NF*E~NN~NJ##~N@~LIr~O\\O~NO~NO@~P^~Q%8A~OW~P!~Ph~Qg~SU~SU~Ot_Q.k`~NILEi<2<Vh~ORm?A,7S8TOCL*~NA\"~NFj4K:*-~NG~N&~OeHT]-`U~OdRSb=YR;~O];csP.SyH~OnKxnHg^~NH~M&~OZj`F<8H~OPJGM4-0~OX~P05VF&;:|KEQT&\'Q~OwbGNKX)No~Oxd~NLFGQ~NJ~NG~P)ol4sG?gi~Q6~LqF5=T)~P8~MH~LYM^??+@QH~OTcXcO_F]HRANm@_VXBOY[aeX~OSMbZTwP~NI5K6=?%~Oc:hO^bUb~N;wC.L%I~OR~NA~NA~Obe~O\\PKIMrEoIkL^(FXQe1_ZUJ6ZMBQP*O~Ox=dd@~O]dKA:E;RF~N@An\\M^C9VlxGc2iV_HZAF^2O~OljB7<ekb~OU.#~N8~N1OIi{2~NF~N\'~NFRNgbB.~N?7Gcv|tTOZo3:\\\\~Ofr}~Od~OW~Mn~NJy~O_ybjQo+|o\\Y\\m^~OdlV/LR\\k~O`o$~ND~NJHVS~O]]2~N>~N=@\\^u~OPPA;STzyYkc^ays~NLZt~OSkk\\\"+=FDO:~N\"~PXq~O`}wfGN~O\\*5DkOG23ULEPZKT~O\\\\XS;FbRW;D?sS[:%k9_M\\a>~ORCBXoNZ~N>S`3L38~NL~OU.~MyGD#H~NG4JaT(?$<T@;H?>DpI(/KZTKJP`~OTw_QSJk~Os4~N(<YJss~MyS~OjGiLE~OfO\'+~N9~N6~N4xG~Ou!~NN~MO~M}~M^~NB~O[~OQT<~N)~N4&~NM9EgD8N@QciFHLJKjV;8]QVo\\Y/YLUX^`;wfQ;KdSMmP~N>^iQw~MN~PT~NG~NIm~Olo~O\\~Oy6c2\\~O[\\Z2CFN;=O@^OGKBIE_L.Q>Q]~OQ_>487SSGDFBiE9T\\NDlBOYWHEQNJQQYXVaPVS7Wde]5S]>-.XT~O[IUA<:|~NNBGMHRFu~OV~QYSKF]Pe~Li~Kj~OTdLb5t~N\'~PuNL<^QIYX_M_PTJNeA]Affma_LUA=>iNz_H;0BL4~Ou>GGWTZ~Lw~P!bLZISA~M,~K}v_SZqf~MG~PcfLLSBWT5Q@@8AEUJY$F&)CDU1E~NKKMXWSRI+HIPUL~Mc~M{~N6Z<_=C~QN~P|~O\\~N;$yXa~P|~KJ~Pf~PW~N<~OnB~NOt:abHEP56<.I>EAj~NK~NI=1.5F?U-,R5_UO!!/CBJX4CGZ87NLI~Q\'~P|nKPzNA~MR~Lm~PY~P\"Vm`~O]2fOHeL^T(~NOU5VY*CCVI?=V:7q((\'~NNNQ<~N0>?(&AB@~N+GF~NM/1IW5AB~NBD1~POt~N9~OXam~Ogk~Og~PC]ZXmlCFqQ~NMVO\\WP@5QD:OQ`C-N=\'77B0~N.<1,66ZE~M}~NE>[;CjQ6~NI~O[ti~Oc~PM~OP`~PF~P+~Pt~PW~P\\eb~Sm~Sk~Q)p0\'!,~Sc~S%B~ND~N#~N<SM~Q%9Q~NF~N8~NCATu~N;~NC~NE~N0~N3<KJ~N,~N7~N5~NC$?MO~NM~N+~N6$HSQ2MD;=^IN!F<DX?`8~S/~Ra~P)|Xd>e~Ra~P\\~Mj~NN#~NFNB~P5~Mz~Mc~N/#~N8DLn~NI~N4~N7~N:~NE=Om~NM~NC~NG~N4~NO7Is)~N/~N7~N?FNKPT8?3[bbXB8O^AdWcfJTQAOYVS<49IK_+`kQR[GOeaVH9icTjb=BbZMRR8@J~NC~NA_iAj~N>~N<8~MD.:0~OiZ~NDg~NGwH`EO>aEM`>Q7FJ`K;_OpXWg>_aTE;M[_PHZHD;H[$WKZHhWTxf~N;Lvu~Mx)~OS~P!~MwW\'wB~OPz=MIIG?YD^YODKK=V3JVWIWQI?JE~N%~N)??@MT~NI~N&~Mt!NJ`T.~N:~Mo]XGa~OX~P$~M@~MF~P%cqL*~Oi~Qk~PH~P$L/BYM@GQ_7>UIUNQ=??YQ\\WIO?FD@~N0~N/JJAKT~NO~N*~N22SJR];~N6~NM6CH^cm~P\"~P=~OgHH~N;v~P:~Q4~QX~P7~O}X~M;~O|as~N8q~Mo~Ow~OVICRVt~OT~MSEJ@P\\\\jT_Rk1.lKRALTA>YHIwXNKXGBg~N:wdGIRE~P_?~N1^DY:~Ou~M78~Me%N>~NJ~OV~R+;~N>^[\\B~PS~O`;]OT]U~NHA6KfAEjJAZQL@GdaTQ8RAWRRQ_K=8@U~Mx~MRtHTHEO%~N5~NK~MAHj~O_~PC~Q2~Q[P~OjoMR~OX~PM~P:nS54Y_6~N*mL@@Wc$~My2MIN)?)~N3QWSDO9=!XpXE>FG~N4yYDVnr;~N8~ND~Mw~P4~Omy2~Py~OS~O[~OjU;~NK~NB~O`zUAA4!\">[FOBPAR9~NKIXH6WB1F;?HA2E94g_I@;N5~Ml}P9br^8~Me~OY~Oe~QT~Lq~R(~P;H~OX/~NL~N4~N7~N;.~NC4Y[S~N?(=FoTQA~N<~NC3DZKB?~N<~N:H=FESZPVPHDC`HQPKSFSZUDIUBb~OR~Ox@~RL~P\"hh&^H~Ow~O]u[_iGQ.)$091_@0~N<~N9~NH<XAKG~N>~N@FC?OMZSbY?PYVMNLLD\\<WDJUQ>Y~NNZU`WO~NJ~NCG1=DWLKsxZUo5N/:~N8>HmQ,>N{2MXfa,L~NFfeEx$~Ok~OU~NLW~OZVn~N.~P\"~M(w0{:j~MB~PG~Op*bg=_(CB<~NGT<>CE~OS^MMc698SkKCL[*&L?OT[W4>85.xiPL~NK7~N7I~Oan0g~On~N0o^mXGQ~OrUmIEUC[~N?~OR:~N=M_FD*~N3J7IKrICE>ID7;~OdSbSKWAMh[=X/>D=8U~NI~NO;M8(#~P-~OkE~NGJQ6~NI~Of~M}~MljFYE0z~N#e{EcNJ~N8~Oa?N_=e`4~OQHW<Y;~OhL;-JRC;Yy~Oe7=TD9pYp~OY~NFZBUU$~M,~N?~Odl~NGyL~Md~QM=~N@~P,%~Otbzcl~LMKMWQXQHf~P<~OSmtaIB+/I!lajtG43BugAZ+A*#MFbmi`K9@@XxMObDS^6V~N,~OlMr\\ZN~N,~Lv~M?fd^a`~O`~MD~M@}pX]M{.#\"A~OTiIhP?6(m\\S^S?2~NH.8P^W]L=5JQe`RXNBLO~NM~OSubrgf~Mq>~NO_ZfXsE5?H>n>K\\3`PTIU-a\"_uHBCD\\)ZEiGEYmG?JJX^h~Od~NN~OVZpb@o~Oq~MbBawXTi~NG~N.:\\2TQJ~OutgP,h>Vh9W9UMZJZADDQZ:A=~OXhSU6KA<~N6~NCYe>D`5~MLI\\^TAf~NHZ\'CBG9FM~N5P/KTB?DM<~NMIH<7JfU90J8,.GFJUHgiE7zGG]~OPbAX~O\\5M[t$~Ov~P\\:~OYf`f~P!~MTbW~NNB@MPE~NI\'0ZT:3Dce1I6T8N@GDN?V5BZVTEE]eB~N7N#F9se&~Os>~On[mX~O_~Om~Oq~LEh~N=hQ~QN~M9+WT\\(5~O[~MyJWZi7~L[1`VgoV|~OV~LKcd?R~Od~Ot~Oo~N4/PGR~Oc~OZe:`FBQG@bSRVNHNHfF0MOCWSP[yQYdo/=OGECZT~N:~Lt~MH{cXf~OTX~M9~LOdKGA~Oh~OY.~Ml~NOEHH~Of~O\\i+sMP_G?Y3o;QW<DdN)[HSI^bRA~M?~P<6$~Og~Or~L5~Om~O^N`F2Vr~PQ.4g~NHB/~N?~N0Si@_d[q.ALeLcA[_s~N8n<~OS]b$Xu6gW9!=~NG~OS~NL!6i~Mv~OuRs~OuKLW~N88f<\\bVJubDFV!~NDD?1CWEYcM*MZC[egfH^f6<^ua9&~OX~NMJ@]~NC5~NM\"f{~N;\'Q~N9}~Of8M&baw~NB8VZPNg~P*~N4:T\\]~OQn~NMKQ[x^P.FmLQbOP~N=;bM[^S~NCG_&;~NJX)*4R?)-HQB~N;~NC~P\'8-~OX~Mk~Ooc~ND~P5TSQkKg~P5~MA<CG_b~NE~OPfT^dcQ`~NA;<AL`b$,SMljJ<8Ox~OhO_#B0U~N,~N1rH4u~N2Y~N70f0S(5V~O[V=EZG6fe?JB[=4b@@R7BIC\\pIWZ@AG~OazS$/$9|XUYC~N9~NKM~P5~Oi~N/g~NK~N>;M~Oe~Oi/7Q;@H2h_BVGTV~NG]MgSWUP4c~PAGc2\\:E8~L@BJNPbA+~Q)W+J/J~N?~OX~OnAO\'^N~NO~P<~MHGU~NM6{M~Oo~Oi2~OSo?Vv;~OU>U-BD+9BC2W?:~N9~MO~N-OHiWC~N<D%+ZyKH.C9Op~OgQ3CpCjG~Of~Ohi8~Ofc~P.~OSQ~P$~Ok[~NF&~NI~OPW,`{$~OT+q6(ZE~N:b@$y$Q~NO~MD~M*27w@Qf|wG:{2M:[~N1H`u4J(Vu|~OQmGf>~OayW<~Or~N2~P-r~N/~Ola~PN~Ol~O]\\~N$~M\'~P=r7T7Mb~NAl34[D[&MUC9INY2^3RMIVRLcOVNcR\\OVHAZLTPNP:H,:KT\\QT~OS~NB~NO~N\'~N7~P.~Oq~NLWYf~NI~N,[~OghG2F*A$dGIG:GBCaG]PLLP[YA@DSVSHV9J8MWSYH>~OQTPIJOMI~N.fKESl~NAI~N*)NTMDw)H]gWK9m=1/O\\^DMRK~NKEZ^IDQ3=~OQ@KTJ~OZ~Q1~P?c~NMa_k)~LG~Oh~Od+M~N>9~NCn~NF8O>TUi]~N>[VKDsDE8`VFWUK/MBSMEUP>ofPN2XW~O]\\~OX_OVZJb~M2tRNLR/~KX~MZ~P:m~NMP}~Ocu~OX~OZMIPkc~L`F`UH<~NF@~OW~MsyoLIHL3=QQdYSK;&B]RWORT1CHj]OPN~M_~NM1BW!!Znc:RHX^~O`~N@wJXB[~N=~LK~P(~OY]MBmO3~P5UZVK;@~M}YmRWJ[Vs^\"USUVGg~NNoVdRX;~Owp157[Gd~MX-W@c?`$nH2EJKSR#QOMFPdWX1~N5~N4~NDTR?X~NB~N%~Mv~Mm~NAMAk~N0~Mw~Mg~N(~N4KK~Px~N5~MW~N!~N$~NEMD~Sy~Og~Mt~N4~NN8PG~T;~Sv~PVlmX8s_I>HVDLA$MOCPUYO?\"~N6~N/~N?R]F:~N3~Mv~Mn~Mu~N@IIu~N&~My~Mq~Ms~N1ME~QP~NA~ND~N!~My~NDOI~T]~T2~NC~N*~N10RU~TA~Tc~QVgEC%H~Ln~Om~N>N5up~P,~Oa~MrMM~NGgDlFEN+64~OSq2G)mUzYNY5~N?TaPD_I<Ad~OSa`RiKQuLS&X~N.tVpW~O`[~N2~LrS~NNd~O\\~OjP~OT,~M:.=H[RJ(>~NA~NJ0CuqR2!NU~OS|si=~NKaVSetr<N~OX~ORVhHnLlrhZcF~NMVY_`~ObH@OeVSNEdEEKUaVEflHCSU~O`{C~N\"ISM~Op~Op~Os~OP~JoOQP~Ov~Oq~Oy~NN~JwMMO_~Ob_~MY~LYSLMXS=7gYaMVFP\\#=Q`PJFa5OPNeGWzVEITX~O[fD~MnVIS~Op~Oo~O|~OS~JtTHU~Or~Op~P)~QV~KCVUOe~OVj~MU~LW@ZF]N=/`ARTNIXI~N,Y*7HR=E~P@69a\\P8~Ow~MJCGaMW;~Ob`HP[W`,~Oa~OSrIaSd7~O\\\\COdI[<qBB;HP\\8!~N)t1KKG@i~NF]Y8=f:g~P-/6dTH0~Oj_DMXbK~N5~Mr~N>VJ\\QE~N8~N,~N\"`N]OS~N\'~N3~M]H[cHR~N@~N/~N\";4\\ZC1~N4~M_PTJIUO(Y~MT~Os~NH!dBNcv~KO*P_U77~Me~N\"~NIR_KNV~Mn>~OX\\{8UaZO~ORvnKIQaUbg`@OP6\'Z>L;H+`4e[QY3\\~K-~Q\'\'QUCDS~R3~N.~N(g[BPU~M;~M}kX[VaM~N*~OU~O^rjEQWmM~OXnfHDN`Pq_m@Y5A6];MDQm;qHPQKdBjnacO~P%v~N2Ql]EZ|@{fBM`gOo~OrP@PT_p\\iLm^AUSO9}HS\\@CJJkgbd;NY\\RmaFZa~Oa*~NLmG[2JlR6xD,&~N=~OZkKCG~N>-~NM~NIC^WHA&@~NJ2\\fhV2~N=~N9WQWgJ7JUm+VWOJGnJd/bXbPG@7oZ,~O}~PG~N+~Kyo^?TZ;\\(90JOo9h~P6BWUAWeS~OcXX`ODIMlSWD]YbZ~P*MT[ACTX~N@+e-mDk~OP~M.VnNEY~OmY~KfOVWDM{9L2A;]~OX~NGj~OVLCWVDu^Nk\\SBFx]cJCENbUd7e\\VC-]<7,Z0OFpN~MeSJO?]I(3X~NJA/Ki\"~ObA@YSW~OT~OftUOKdk~Oi~OVHP@7N^_~OSR5U\'&~ORW.\\U9~NA%=&~MbA\'~N+,KS\\~NHZ2XKY7i~NHE2Q1I)~OX~Oe~OQZUjB9Re~NJC@bUBQYtpK5[Y~NK~N2~NI:VJBb{~P1~PIp*DbD^~Mz80ZQ~N3~Oa~Mz~P\'~MyF3~LX~L5~M9q~N:\\b-0~Mwi~OdG8<Bbf~O[XRYQvoko~OZMKY]u_rX_\\?KWnXI[_i9K[DI??EaVSQXO_Kd~K}~Ls~MWz~N?KD@=~Py-~OS2INESm{OSKOfrmmtCLJgqkl[eJ3HblWUSdmASUPJQV~NFY]TPQMF~Q0}-H\\9~N1}@~OoFIJH>\'l~NC:JR>4CpCY2+DN^sL~OiQ683[8F~Og~P!pK>[;6LXT.G@q%<WFn.~Ov~P^W4eCV23~Lm~N4:nUaZ5~OXBW6LVQi~N=a]?HGV;~OZH|~NI97DiBW~Oe~OhglG\\~Od~N:_PSWl;(a6y#w5,~O];ZNzi*~PjQI~N0/^<~NI~NK<7&<h~Oc^t9h>2K~Os~OS~O^EUI$J~Or[OTI~N3%OiD`qi~N=?H]U7~P1b8ROT%~OX~NC4@RN]2~MfF[~N8@D~OV~N*<YD;JP~O_S~P,A{8KD~ORUG>lK]4~OZo^y$/C>~OW-3~N6~Ot~N2XLc3{~P?~N-eMG\\=?+z~OQ~OUltm\'~Ofa]XJBY~O[xh\\@9Ijfhe@~NL~NJ>at{eD~NK!ERr~Ob~OUE?E^gc~Own~OTXV^W~PI3~OTi~OZfL~O]E7~NLy~OQu~Oq?2VsGX\\my>`N?1(`~OS~P+[/,~NH#5@~OZI3,~N>~N8-OX\\L=~NO\'0/YLHGLC3l\"I5Vbh7Q9h~P/~Ma{&#~Ox&~Ly~On~N=P~OS*~OnM~O^{4ND,Oh~NA~N5bIRjdAM~N94=OCg~OW~N-gcT`I~O^TF:N=Z2TJI|IHQ<e~M|5~OU~O]?;,6T~LxL~O[L6o~Ohr~MW~NCIXX/OXP0~ND^HgmfMA.bH|ygCGnmtIwB~OX~NMTJNUKS~Ol;NT[Af~ND~ND:cUDAH_]ig]3?C>[~Q.~P$KDU>\\M~Q9~PF~N=~NDOOCH~P4g~NO~N<WISHT$WEKS]t+3KYDCQe@HWeOBL~NCEQa;HaO~O|CeZ=M?e_~P_~O`A?Y?cq~Pl~Lq~N?~N@9FB>~P\"@~N;~NERBO~O_LUR2N`__T+YT>VLNCWEV[ASAdhGcr].>~M{QPOE~N<GJZ[Tj>9CTTODHXYFsNgWM^HP7{y~OU}x|L~NB&~OuYJijo~O^~M<~N,~Mp~Ma~M@~N@~P3~MSMY]b\\=~N@YOQ@FR3iSmcQcGE^IMOY`PTOWN\\PZUJB=~OVNgm[[e`CbhqE~O}6~LZ~P%~OtEW~NM}~O]+~N;z~NO-q~OX~Mp]A(7~N1~OUL~Ojb[VHI?~Oj~OhcykMJYz~OmVGJO5bIGD-K[TS@:~NOrBId\'~NO~O[S/~NG5K4]lm&dR%~N3~P^~N,@=cFW~N(~P0~O}vloC#^G~P4s*L!3~NB+;GU~O[5AGvrOW?\\NTT(~ND\\gWp6j|u~NL~NBPKAo~N7~K!~KF~R6~PZ+_AZ~K/s~OcWVN\\S~RL~OVTMNCEE~P>c\\@PEMN@UW<DPTM_VLETBHVCdAJKS\\O]J]YNY?G~Js~K3~R5~PD<[ZZ~K@~KR~OpPR_]X~R;~OT^NKCGL~O|\\^BHJONFQW;EWXI_\\YEMANZI\\MENPUR_WZMGQWCC~NHr~NJ<L]IA~Od`F)g1Xyx|;\\[mRcZ9~NO~N5EPL^H~NO~NK~NNESA~Of|r7abSmo~OR~OQGU~OWlY\'o~OjS]g~OQN~O{Y~OPD%~N<\\{~Ml~Ow*MFa,!~P(~O\\~N=@2>`f~OQ9~N<#=/R*~NG2\")\"~NNPO~P(>~N<~NN~N06M~OX$~Py#98o~O]M~OY~NF~OjVF~OX.qK5:]4~MB~O]~NIC9IK4:~PBoF8<=Q~NFB~PuVFFKC~N$~N6~OVEMCED5~NC(YAKKWAN~OTUNHVNK:~OPPOAWHE-~OeaKId1~L|~Q1~R%:A8RD~NI~P:~KK`AC/N~N0I~QOAJHFE~N?~N=~PjXD<QJE~NB~N<GKQ;MV~O[~OZKS7XQ7IjU7iWH5~OS~NO~N?~N?D*~Mu~NM~NL~On6\"@BB<\\~Og7iRFVJt~OoG@;,2~NJ?@Y>P<J)I~NKB>>9Cc@`\'~NM]i9H~Ofn@?NV~NESq~P7U=~NH~NK~M@~N(=[$~N5E30Md~P(~NMO[S9N~Ov~P2KHSQG]~OY~Ps9H5S<Uz~P|/57BHHj~P#M~NJ>=Sit~PI[@(N~NO;~OX~Oy~NC~N*aNOd)F~N)~L}H/-U~MI~Myg^=C?B0aL+(u{4~NF]=1%{~OSD~NLS_LOIJrH[3~M.I;4H~M&~M{\'~NAMMKd~N.,~N:[9>W<\\~M6v!W]Sb~OY&AhOX;Q[;OP5~OY~OiM@#CT9~Oc~OUR^X0[RgNl^.Hc\\hbifF~MBU08?~NOD~L|q~NH_|OekR~NH~Lo~L1)7cN[D~LL~LF~NJIKR[x-2&:PXQI$DLTsP]`]GD`RQPUa^]OTgJn:FM<POGf{fY\\FUoo~L=sVA4labe~O\\hC>5QWAnVFPZRU8OL@l\\e@A<CVaQcWkCXoLUE`CJ_MbZB~ND~N$~NB0:3~NF~ORC6>Qb9AgjG7~OWs]TS;.FciuQD:@XQbsYVAMJM_/S5zx[[TM~Oqk~O\\@5OK~N?D~PG~N,-F4D[~N=~N+~MuJ~OQcTCu~NEx~OQIMI7IfmLp17RWR4XuJ@Ll=qhljpGYSp~OZWJa_[~NK/+w/bQ~N8S~M1[SEaKRGS~OQnS=U9UEoiP?N\\O>TN~PN~NM*G^OE$+~PHBPTD]=%MUtT]+R>Ih~OgdLRRPZYX~Lj~Jg~O|>O]CBy~Oy~LLqUHHK@*l~L<XOCN@JYc~LZ~NM4@hZ>7)~LUTQJMS@6`~L,~OW>?4KDQn~LZ~Oxi@9aK@~O{~Ja!~NJOTM]7VD~OTtD,Q^~NNl~NKj7b@^H8~N9(KE;>B~N@~NBZR[PF^~Mg~OxK]\"Y]F~Mq+e7FVQ=[0?k/n+RCn8VaU;U~PI\"*9$@8B~O\\PHNLYFN<V`[[I@BW~OUFSYI;K~Ol~MzOJ~NIPSQn9j=XBM[m~OPE\\`?$B*^d7~NJi1K%Zn8dH{)R~NF~N=+ViFmEB~M^~NE5ev>0J$~N?:qpCaY+PoXJt^~N(SmE\\~NOo~P%=^CQ^M+f~NO^b:Js@zG>E~O\\K(~Os6~N,F~N>TnJ:,D.ubHpXES)~O\\_LYQ4\\j.~OTeQSC^Nfk;{Y%^\'e8Vw\'D_DNObb-N4MFMJKZ^=B]K\\Ve9XVCC;J;E7~NNC0WO7mY~NH~NFTh?HZo~OX``/gr~Mg~N@~NG~P;+~N.DS~O\\UiP6hO\\X-A<VWUeYNGH-MTaZOW^`(JH\\:\"Lac.L|%S~ND@_NLT~NE~ND3VGfL~QOA~P\\~N*~Ocdr~P\\MVJCAG2e&=CPJC#u`E1PJ/:S>R7~O]~Oh-2&7D~NEz~OZ((4VJ~N\'\'?~N4:~OVV~NFLNW~OSeL\'~N\',=H~NM4~Om~N?XTVI~NHl~MV<T0g[@N~P-5Z3_R+F1Va=~Oq~Os=VcGQ~NK~Ox~OxI.<@~NN~N54R~NGKb~OS>6\\V6<~LJ+6*\\Y~My~Pt~PEJoQYIg^~L,G\'~OS~O[h@~Mze:XzjU0/\'hT?K~N@HW4V54,EIkWbML1OzgWFT<.@N4QzLeMY/<A~N0nLdL^~P2~M&dh~O]if~N/~P?u@ue1K~OV~N?~NOpHN<xWHRBG;Q/&]:[E~OWISsz_Z~Oz:>Ig^Q~P!XXW\\J\\~N%k?aXDII~OXDWB\\J:DYJBCuQ-9G~NBEp46DH5`[DPFsWFUays~Os~OSQgV9~OQ~Ol~On{~MxW~Op~NH=r2$=`BJ];U;Flo5elYC8w3VF~OdO.V,FHsgb3AS`cIJIVE\\\"D%;/~N=C2kT~NE~MY~MK~M\\;4Z~OQF~N0~N!~N+~NGOJd~PH~S>~S4~Q}R~OPH]~Oau~P4~P?qL_O5~N1~N*3KcJL~N(~Mn~Mi~NKOAB9A~Mk~Mm:HG5DEWTNJ@[BAMBL<BWIJ=@GKAe5~P7~RZ~RY~QP>~OwFm~Oi~Mn~Lu~P?jRcN8~Mu~Mx7dW\\M~N/~Me~Mi~NJIII7<~Md~Mo9AP6KENHHJCRLLLQCSQPNOCEN[C\'~M`~N/<A~N1~NM8@.~P!~Mw~Mo~Ow2Jek\\~N2~N2_gfBB@~N\'~N40>FP?.~N0~N56?H,.H8P?77dPIGV@PT=bM?AVg>.~OY~Pw~P|~Pv~Ps~Om~NLA~N6w~Ow~O{h~N$RaLD:*5JS[J*\'~NI~NH<NiE6~N:~N9:Cg~NL<D4E3:3LUQ5<EZDSUFJAATen/Gy~NB~P@~M{~N0\"[*<~N/~NA~OwhYk<~Ou~M|~O|~NCWGYO(~NGJScKF~NF\\F~Ok<5c7QD4=\'DEmBi~NMiua%oBP7u~NO~OQ~OY~NK\"[~Ob%~O{DCW~N$-b~OVLeK[O!jC$~NK[;9X}j.MS^3?Aa<n]88iZQ?e[CGaAX;`8KE;H]hl_~NK~O^^[z~N)_~Oc;;=3@O~N&ULGDN~NF~NB7Zh_C<3~NN%hX`]\\[>i>0SF~OPw{t\'iW,ENN?H~NH-IGc]~O`~N<~NG:~OY~NGu~N=~N>~Oog~OZ6p>~NK~MCaQ_+k~N*G8yV]ZSnp^DG~OW=[?~N;``?KFj|~Oda~N?NP^.ch|[D3?mM_9PFU]KE<VORL]XJJqYXJ=[@RMATq~O^~OUhk?[>4~O[~O\\EC[HZA~O\\~PC;]ZYl~NH~MJ~M2~N4mEC>7CM4~NJ|=iAPR5^^P_PUUMJBOWMKEQOLTNb~OX~OTf^RKNAx~OYZKPK^R~N3~N/<kO[4~N@~Lv~Lh~Mi<TW>ZQLT=gUCSIMIRdMFaE@^PG;Ol3hTQFD~M{T~O]~OTYHM?~Lm~Mu~OX~OSUS]Q~L{~Mpl]XTO8~N>2BdSB>{FkI\\a>XMD_IK:T:PWcEL`VX~NMS_@]Q?G\\~N0L~OS{^PN1~M,~Q\"~OR~OP\\LUX~Lu#pSWRZ?H~NJ:]TG50vNQQ\\GJ~P4~MW\'J~M{~L|~M>~NLLCn[QL+CB]L~NG5ZZ!EJBUOrgKBRgNOMQUJLJXUJVIQTIYDI9>NJ<[R_RR~MzG~OP~M)~Q\\C~MS~N;~OlZw~N7>~P!~Ofn3GA*;B>~NLPOFbMjt``QVL[>\\LEHZLRJM9KYSNONFcM4ZZBRL9bF^=~P5~Ol~R3~RIVNB</>~QY~R7PG3~NI~NG~NJV~P}FE%~NA~NO~NF,~OPI@=~NF~NM~NB=2MI61\'8@]NQM?BMFLJJUERBQh-ZTW~P?~PE~R*~R2ZFC41o~M*~R2QF8~NE~NHHz~QMAG(~NJ~NN~NO(~O]FO6~NO!~NC5BME7).1EBJ`LANGEO9NPGFTV7~NJY\'%Pyg~MDj~O^VhKP}i~ObE\\LvOl`%0O..y1~NH&~O\\~Oa2A^k_~MhuPRhL]~N=~ND~Oku[_[iREKkT<@Q~N@\\E>:DTYW;/C?al~NE~N4mZMwpKIbQOGVU~OPB;~N3=~O`NJyl$~Mc~N4~OU~OVr\\>~N/~N\'V~OTeMJ_\'~P#~Oa4ZAKYY~NI~N;KqK@~QR~L*~PLgS7Zf~Mp~PE~N#~N<`Bn~O\\~Oe*Q-hPWJdU,~N9AJDS?f^AGJOLOY<^i9a9{\'mLL?D+~OV~NC|k^pbXa~N.9IONZ2~MmN91;O~OveP(/~N)\\ZZ~OYgUT~NH*VeWa^h!0]GHB]~N>BI;;L9IA~N8]99A0oUJNK5cf~M4eU?DF(#~M51O?.-T,~N!hCG;7<8~N4STJ82~NAM$IQELL9~M8KWk@-_K~Os:9;XlR]*~M%Y7Tg5ZPR~NLO@C>91~OPWTB3A:$nXM64D+@~P2]@B;A3O~O]HWY6HE~PE~OgPDO.GU~NH~Q6{:?TjR2~Mp!*FLI?`/R,;>LRs~N%SB08CK~P.#Q@X53r~OybOIW?:n~OyZSAOAR6~Ot~NO209=N;U~N55:ISM>~OTE/fQ`DUjsaXH:OW\\~PkcCKETM~OT^[EP/6~NI~P4gQTT,;2~P1hiVe>h=fs:-MP8>>~PG`R>SKSk|~N/l8YH4H\\;~OrJeUcB=iv0[\\ZGC~OQP~N=z_kHO^d~N1~OUgdRP@X~NLDD`FD~Mv=?]F^:L~N$~NMaJO3VX~Q9NSRI7MW~N3IUUVeP:U3)RZ_UD~N$~NO~N!Iee@[~M_*~McVk^H\\~M3/~NG#JiW?~MTFF]GfB2~Mw~N@V^IBQY~Oq~M@~On~OTCgy[@Z1T4XGPN`g=SbOK5aXPFK``FV:LKgFYL6@992F;:?N+>A\'Sq*S_~N\"S\\[~Pf~LE~P<~P1~My8mo~PV~M}~P##LmgJ~N#~M{TfKY:MPbZMJVVKJV,ANm\\K~NI~NIKT?@I;~Ok~N+[1P6Ig~P$~MN~O\\~Oy~MSJXL6/.7Nf~Ocn~N?9~N?~NNcu~OT~OR\"~NE9:;cD|J?W??I~OQ~P-/-.<4Uv~P88<?^+Xz~OZV(~NJ~NF~NNhh~P19e~NB~N(~N:PhEM9~Mt~O[:~OT~OP~P3~N/#>BJP~Oi~OaH*0>^4h~Oi3\\sVMLa~OU1HA\\S@8~NOo0@eXO^a~NI*<C5D~P7~Q)Qu~Mw~NA9~OP5~Ow\"|7#P@AI~O`D7FsVf~NLQYL~O`admr~OTckD.Bn~N<$~OeIFEnc~OVB7cJ_W]YPb&~OV^IkgXxW+\\%~OTa=CZ9&?Z$a~N;\"\"~NF~NM3+Pa$RsW&ZQ\\l8^eL\"FKXUE~OiU4;89^`;4G^\"6\\P-(jnSQ;-VfeVW8EOIDqWb^AK6OSIR])~N;~N=\'hF7(~N/~N$~N7.dFD~NC~N$~N#~N2~NB~On;E~NE~N=~N1~Mw~N-~QzV<;6~NN~NB~P$~SSl~NOQ[~O^~LL~R|~T6JQFNF<_=LY\\EN/b1CYZ\"~N?~N<%R?@#~N3~N$~N,~N<6G:~NH~Mw~N,~N:~N3~OhGE~NN~N+~N:(C~S-L@-)%!~SP~TCZ4CI~Oq~LT~SI~TD\"~N9~NA:k~Oid~O__IT~N>[o}n`A.9Q_]`#sT87DV)~NKrZWVGJ7qseR6JLtxT6BEBYuzggQqcTU3~NIaQ~OkL~OWn~O[~N>j7V~NH~OV~NC~NE&k?o&nE=4UMW~NMC@=?D<B?I2r2YWD3#]usSL@CuIe~NHiJ6Q6~Oe~Jo~OhDNiG@}~Oh~LtyYDKIHK|~Lq;PKZJF\\5~LF~N7LEa[GP~N<~LF,QS<MFM.~Lq{F@@MDV~OS~M#~Ol|GHgS@~Op~K!~L_WUPaO\\@V~OVw^B[G]Xw~Ow-DIcWGQ/~P)~N14AgVE@~N5~P%$TPMNF@~NN~Omr\\`<VIUi~OjVJlW[R_T~L]FH;0G;nD\\sg,Am~O`D3dFI9:y5-&EJKX9]?B8[%3~NG@IoAV#BS]n~OV~OSL~NF=[R)M/]~NJMJQ\\FR<7OOZ]7WLFh@M]O9P<XU^CBNZc+B>F39d~Ob_85QYQ0X.~NMZUSNV=9NPZEa<4TJWr[NY@RC`KVVWUPOJG[cNS\\iYTdoXHofP7\\sSTfSL~OyTsldEVO~M$~L$~Q<~O[kp]a~QG~Jz~MD~N\"Km\'YbQHVBOK]TMYU\\F\\WBadJPSdNUZiS[lWSNFeaaqJD~Oe5WcgNVJ~Lq~Lq~M6|~OR|Q\\~Qh~J6~Lo~N,Hm<k5_?Y!~N6h~LaPBR<T+:RoXOG+~OWPNKe^d~Ox~NKW`ZT@~OaqP9|KA~OsDb3IFR!AOL]ZS~M+EN|jtcPEONQ)~N,1~N%h6hWD~NI~NMcUiKJ+~Mw3iAdkV~N7-]SMb^5]ESQM~NO~NB[a\\UZ`k~NEJblUM~N3E=P[}kS~M?e[AR\"z~N!|c9!;TDeD>?H4%NRb~NMA~P(~O|_29A(W~P(~OtKKiY-(P>P^2^#@I@q~P\":~MIz\\0]P~NI~P%~L`Dj#;WT{K~M3=&8`~N3~N2fH)\'C.LQ91,~O[~OZ,-^3.;~OU~Oq?0G~O]3~NJ6Jj[4E~N\">+?]z{<~N2W6MUw~P=GPTSEZ8~N@SQHZWJVFFFYb~OP~OY~N$~LXMSW~Oa~O^~Op~QX~KlWPKz~O\\~OZv~KfFJXT~O\\s;~N-USG]PL`QLHXNJT>8YURTS?V]PSK^XY)>JPVbt~Od~N\'~LVPNT~Ob~O_~Oc6~KLSOK~O[~O]~O]~OS~KSTMXT~OX{F~N.UPOaYNTTZMWVQK\\e~OaVCb3?/F~MbINfax>5rcQ`LneD*+NjXZG@~OP~NI~NAs\\j7S2~N1$4Aa^5~P0~O^/>Jz\'O~N;(~N:i+~Os~Oz~MkR>6[Q:B~Ot~NISh[`\\,4D!awRv.9y92nqOQ;T~N3~N7QnnDZ+C~N*~NG^kRd~Ow38!g}~N=!~P\\~P?#0lf~NO~My~OTUCN\\BNm^,SK_Q?LIAp4~OV@XVKFG~O\\{~N<5UVc~O]~OW~NC~M>2GOL7~NK~MK~Mx7TQ;V;IN93ITLU[M2~N>)GHONPT7R^SMgLG7OFx1~OZBnJJFH~OY~O`~NA2Y[a~O\\~O`~Pm~MdD4MO?!~Me~Mr?CCWcIG>3R@>=OI_]3~Qs~M!~N;;$~N:~N7~Pk3$~N<2h)Z\\~OY*~OS~OYY}VQZf~Oc_LI3EBN[EFJXCjH96R=NIe=?6G^IX%NBJL;^~NE~N!~N$7~OkKF9~N!t2~OS,LkjnsY~Om~OcgUe0E6~O^~NO6GD[mAF95@UCaBO7R>SOTY:@LUZ2Wf:^QNJb~MSFU=^-_?x_~O[&\'>MW~N=~OjwUK=]?~PhDm<LW4f~MoHeEI;K4z>P~OS?,DWKw%AUW{2~MA~O_3HdA7~Oc~On&CcTQFO~MtZWG/E+%~MpO~P\'bF:dL>~OQ~OghOO?VxG^dN~NK9N9=B[P,=B~N<~N,R>CZY7~O{cQE@A[EfQ]frkFq?as^S!~NO~NFZ/QR@~N*~M/*ZH.R5~NI~N;SKQEH8KAg;(\"UcNq~OVJ7NZhj.Z~NN~Ok~OQj{v~Omcg~OY~OXDN/y~Of~NCE;SJ~NI~M_~N)nPH9Az~N9L95P:Q~Ocf0y]B3AKJO~NH.ZPD_L]NNS^`si~OZ~NOr~O_~OTfq^utTyUC~N//~Me`>z*]~OV}*>9-`~NJP~N53O>kTgDx]l:~NI=LKY~N=PVh0a/~OV<n\\P95nG~NM8.vGf*~N7;~NN~NH1O#~NM~P,~ObVVU8~NF~NF~N\'~P=T?18~NH:~OPrL5LXAlh~OrF9Cj__\\~O^<QL~NJIKI~OkNO@N;14_?GV?,4M2_d;PQLgW?eCOSFT:5D=C7CGUAAP:<C<RRLA>88PToJ>~N8\"Kf\\3;sAI~OQ3.~LG~KZ~M%~M0~L{~M7~L)~M.S+?NHM4RMj[FC]VN;F@KHOH@ORHE8CV[MIBPD9OR~OXF;~NE\'>[~OQ1~N1~NB\'6~N@~Mz~NM~PL~PA~P<~Q4~Q&~P;~P;~OaxJUQZZ76V5NWTW~OV~O`iQQh3MC=WLm7YeP4P^;MT~OZwiAJ5M\\_\'8KRNAW7;7idE~NOY]iK;wXqVV[F^~OQhZ_aAJUrg];[bz[UwB1~ND~N6TTPB:+~Me~N\"UB]Q<~NB~Mo~MU~MzYGiI=*~NBG<OUrTJR~Oo~P?~P.%~OR~Oa~PG?C~O]~P96E)=~P\"H~N%~NLC5~NOS4%~N7H85$G/\"O?<+~ND5U~NC:\'D4;C2H@/L3-G,~NM~NE:ISI21~N:~Or~Q=~Pq~QE~QN~Pspu~Oa~P@~O\\r~O\\|~Oi.~MtOWAa2.N~N9:X3?<L~NC/HF;LJ3~NC~NI(A2JB,~N>5~NHY*NJ~NO8;E-:S4~NI\\Hgn~OuH~P,o|~N.fR)50~PCagB?)+~NM;~Oo\"/8#724o!O\"A~NI(.y~N5=_!/~NO42~N(~N(=8GG~PB~PA~M^:01~NG~PE(~P5z;q~Owz~P}~Ogp?~OQITGn~PH:$N7\\<^i~Oz(E*KQCR~Ox,8B@VRh~Oq~N,!~N1?:d~O\\~Ov~NA~N&~N91Hp~Ob~O{)?0*<~OY~OW~Op~O`~L;~Q)~N#~OZ~NCP~NM=~Ms~Ps(&>YKnVmQtUd`9\\m6Bj8AzDUJ>Phm\"~NLRQVDbI]>SHVYT\\s$~OQ%sBAH&~L_Q~Oei?K~Ol~M}~N1~N3O4+;Q>OdMcLRJ~OSWePQeCXu9UTTsNV,06PYVhAZ4AHQPPO~O`JKdQ7Ok~Pl~PU~PE~PB~Q%~O{~R(~Qo~OW{I~NG~NLQ~O_~PN~NBI-5:~NM~N7~MD~NMR/2>16~NIk^G@3>:GIdO9#DG_S5_:9Y-<]JHM@?N8~Q_~OomG~N>~N=jd\"~O`:Y?~N4~O_~O`gH=+~ND7~NF~M\\-Y5!~NF*,~NAjNL~NN~NE/.7Vh[<~NG5I^DI]a\"950,HYG:AH3u&@ZqS-eO%5HLH~NM+@gHYTBYR#Djod_PC~N3DttaFI;6TU\\J<N7)~Mu*|v~NE/)^~Ma~N0~Mp~Mc3~N2~OjY@3[ZXhAd1SAVN1b4@>gZZ]7SImbjXWH0SkoiPaO~N>-T[G=V1~OV~O_<?7a~Odm~KTm;~O_W~Op@~MA`asa^;^~Ot[lpTT~MI~M4.\\cSZb~N+~M[#UVfgCgR/YJKIaMHK4~NI8KJ^]W?C~N>~NB=Phr;~NC~N4~N5\"/oZdKo~O`j~N=~Ot~MJ`{o>E~Ok~L?j>SNS_~PPv~N:cONg~OUe5SMM_z[F?hW0V[<R[c>~NI&A(1tc~N=2~N+~N@f~N:g;D~OX6~O]8N~N\"oUM_xZ~Ma+~NE8~NF\\~Oy<?4hUPlh?XLdP&D*~NNMgZIaHMq8~NG`G-ahYL%`E`]WBC_Xg~N>|W>~On~NKu,PgUa=~NI:c<~P\'~N=~P8[UYnIQ~M{~NK<K879Y*7=QXZb>/5qDbebNImV:oHRk7Z&t\'v5ac0%n~MvJD)BJK~L|~OQDDGU<=/MZ>[kH6?0KQMgaU2iKAisbM0n-51?AEK5tIURde/~O[*~N!QJR~N0~QM~L,bU]*u~NJ[~Q?F>Hz>X\"FZF_TeKR?I\\K^Os#jLehs^OgTTSVO;b~NC~MJFQOQQSSg-><E]>RpR3bnbSL&@[`QrVd,IM:RUMF?3RoKK[J7T+J=[?Pi~N4~P%~P\'~O\\rb~OS=~N1~Mz~OTo-0~NM~N,{Q<JmAWxR7V`ebK..;_f~OR_4?8OORIRH?J9uYX=)cU~MxjwQ3~NBa~O`\'~N3G~NJ~NB~N3~OR~Od+~ND~N=~Mx7%~P\\D.~Or~OQnY<l8iBi\\\\D-~Ov~OQH?iQIYlb=6~NGfhs~OVUQ50lXl];IiR1el~N1-h{H;C/~OYW~OU~Oa~OQ~O_~N=DM:9l[Zn?~P.~OXu~N5Dc#JN~NO+*~N3[A>D~NMC~NL%,Z,g,~NF~NJ!6MR.~O\\a~NJ<bY;O83\"NN%x/~P!6bGf`6~OdD/OM`~Oh+6AjOL~N<~Oa%Se4e[p~N*p4*\\IM~Mz_rqTZM~N#w]#17W~MocA~N9fhn2ok~NGe?QYU[~NJI~OQgfmJ21b7~MzA~N3>AM[H~O}C+<bF1~O|~Odd9HP=~OcT=2=L:~OaH)8BR6~OVa~N1~N=yDa~OV~OQ0$>`BmYbOc9C~P%~N?KX@9_~Lg~N@tr8^4~NO~Otf~O[U[DF;vmVjG^.R5Li^;T-WL=y;.,~NOv~NABW~Oh0j/FNe(lA_5h+~Ot1(JJ~N1:~M(S~OVQU+G~Ms*L|~OhCF~NHJGS}q];-J/VdZKNTdNdguE\'GtFBpsX:WD~O}RTidKCD~Op~MUuL[%H>~NAg;RGOb~O`iTUTcJDK~OPFVk@LF^PWHIG@qH`NVLGpPMcDzRXQ@~PL~L}~N-OZ\\Ou~ML~P?~OV~N4GX@K~NJ~PD~N?~NKNJI:Z\"GETcJIF.MV_SWLHNMHJTIu8KXKJCr@4(~NLV=IN*~P84~Oz~N@6WB%o~RiY;PO@XG0QGBUYOWLq>3F^dXb]+FMEnRKT1B1MTZD~NA~L`~LdCICQN~Op~L`~K{*\'MRS~NJ}4j_mVd_@GYPcI2<c7NZRKO`63ENfeK^?JC?_T^ZVnYIURR~N>~Ob~OXeM7G0~Pn~LecA*9q\\~M:~Pc-h]UCm~OaPYa?~OaL{~N6~Mk<L[Od8p:l>DT[T`~Ol~O^56@KJR~On}\'%C1R~NO+sMdQ>e2~N6Pba>oA~O[jRkH~Ob`nXVXqyf~OiI@&BHktN`!~MC~N\'-IWS?~NM~Mi~N4@H=]K~NJ~Md#47AOX~NJ~MX~N5RTCYT`,2LUiW`~OW~O}=ekH}S~M{=nl~O\\T~OZ~M&QHP1CE~N!cP5fPDaaaNP3fCY~OQ~OVQW1XdHNY0MF]IGg~P#adJ?CU~NO~N?,UXRPeI~Ml~NE~N>~N$dE~OZ~OZ~NN_g2PJo}~OUN:W^)|]i8TF@X[BvFLAaLZ2LAZFe<E3Jx^O@KaIC8ZWS;RY~NGgz9.zg*~OfOc[~OX~Oj~Oah+^gP~OpyP\\<bW_7Q6E5Muua?~NLC]jG`~N>?16^~OZ~MO41m~NNA:~N8t~NH==HNWirCcSTr~O]csct~O[~Oc~Ot>osk~Of~OSI(2lalJ=~NHA]WcN~NBC~N(1Wh~Mg~M9~NM#*1BA~MV~MhAS/6[~N5ZKK9@2PNG~NNH~NI~N?&>~OZ_WLH!$Aa~NM~N=0\'#@4~O^b>FIeI+~P&~OQSgaQA@d~O`eBjz]B~PCPA~NM>]Ep`/~NMMc>4f~OuKp~N7f~N$\\:$=qwz58HG<0.-+YOtBqtYuz#@~N0~N;8=@!c~P!V~OP~OPh~OQnN~O^k;B\'Pwed,W7iI/~NM~P@~L{;ioh~Of~N6k*IMXx6wV87\\RkF4~N4M&`LIGGoTKKMB`JZF`_3P@V4O@D\\LE<j<d6g<4D~NJ~P#~Oc&~N0&7~P=~NN~OcQ~N4\'Q5*~N7l07#LHThX8<JJaNg[cGH9XV~NIKbWDIWIH~NHCRV^EF4XT<<QSk)4@5+?.>%~NK0WH?~NI~NO>CF?dPOB3SQU]][-9TUWaQW~NMJBKhNcIA~N@~NKLVU?~N>~NL~NJ~NK>4~N@A~NA,~M}aHF%@d~M}C~PjNHGG~PX39:P<TSQG5L5UUSE3?NOLILEH*RP_7VV9x~PHP=BV~P_E~N!@124V7~Mk~O]~O`xY\\JQ~On~P+~P[pJKGD~OZ{~OQOg<SH/~OcV|[J;LM~O\\.DN;tWR{M?+tt[CTlDH]MIJ~P(X~NJHVBo-~Ofz~Mp.W>:{~Oh~Ou~N0~NM\'U$6~Me~Mq~N8;![Oa1~N2\'19?_Eh~N!~N15=|jZq^<=t{g`Zl\\V_PFIp0THeT$~NH\\.g;QI2_1~N?_2OC+E~NILc,=:JKvY~OTTCWJK~N57vBJ,,A~Ol~N)jXQaY]~P+~MUkfZ,C8~Ou~Mi4(fnV\\XV?fN,`P@fP~NH23F4Mb8~N>CC3/ddr7JE8Ne~OinHF)4E~N%~Ocsw-K@Jd~OoLPO48B~OPa,~OP~OVA`b.*@WBMn~OR5~NJ{~M{BFHm:v2hH:Zs@%3~Ou~Ok~Oc$o~O_`~O^.n~Oa4k~P/}M3gt/~ORR.-~P(~OSD1~Ozud]idp)|~O[v~NJ;\"b.lJ~Ojm-;UA<~N;mlMh~Os5)O~N:Gz~N\'v,0~O|d\\-3kln~OU9~O[~O[_k1Uvw5h~Oi=075jC~NBu*rZToY|j2~OZ*C??oZ!4q~P-j\"Wa*}|RK4~O\\~O`J~Orj~NBia4_t/5:u}%~P;N~N(~O`~NOn~OQ1~N2.>dlJ7*KJ~NFc~N<1*&p;^~NI?hC~OSsC~Of~NH0~O]xjq(~OY&<~OV~P1O~OUu~O^ao~Oru~NI#R6fyx~O{~OR~OQN(&</~Ob1jq~Ow\"~O`G~Oe>c~NG~O_p>~Og\\v|2BA~O\\~On+~PJ^>T:gU~O\\JUnj5}oaD3~N\"~O`AV2~NF~Og5!g=C0*`,\'g~P-ZR~OYT<~OZo(:~N6+>\'2~Oj5i&~P1x_=~P.8_*~NERz8~Oafa~NC^kA~N,psYx~ORW~P9~NE|j0ZHi^3&2]R7^.0hP~NE&^~N9<TQir];~O^&u~N?~N5f~N#rF~NLd0~OZ~Ogvh~N-[~OY/1~NB9v~Om~NKL~N9~NNpb]~NC#[~N46~OT<[v-~NN|~OZr~N<*y>K~Ow~N>j~N=Yv~O[~Oc~O\\AJl~OrGiJ~O\\G5:es.L~N7ao=r~NL#~Od~NFx~OXY2~NKCM@Vp0~N;l~NM+\'Q=~N3~NH~Od~O[~N8b;~M}~NC?H+~Oe~Of~OP~OQ{~NG^GO~ND~O^)f?e~NM~Ov~NH8Tr~OQrc^~NH@~O]~O`8~P-~OqLPP~OZ#~OX~NIuyp%~OWB/~N;~N:c~OZ5~N&I~N@s3>H7\"EG~OQ81c~Q)3~OP~OP_~M}-~Oj~NI]@c~ND8~NO~P=~N/.\"~O[bb&~Ol$Z8[~OWvYfR}\"(pEDb~P*g~O]W~NL~OZ*~OUc!YnDS,6H~NM]~O\\~NM~N1)Y;*7X~NH41~OVo;TS~P-~Mm{~N=]*&M~N1~NLn:!LM~OU\'ch50~N<~NJ|~P#kR~N@~Mb$jY~NL9_%+<~NJMB~N@9h_p~NM~NGT~MkE}~NM~O^mwW}SJ~OX~NApU~OVWjdOqb8~P8~NK=\'~P.l+~Or\"~NI~OUEfg~Mk!+64Fg~OUa~N%V~N08~N@V-~N<~N;^?~NOsAz@~NK$~NJP~O]ih]~NG~Oa~NJH~Ocr~NA`~N4_3~P$~Oh\'Z~NA/~N@~NCS~OZ~OW~Ob~NK9~Ou~NH~P?rtI~PL~N2+~NI7~M|~N?<hN~Q[~P(A~N6N~OQ~N@j~N\'.,~N&B~O\\Kq7m#~Ms~PB~NL9~Mr\\aUdT~Oh~Ox2~P?Fz~Mzv~Oc^~NAF%]oa~NO~P$_&~N/~Os~N;~N6[Wq;~OYq=z~NN~LIz~Ox~Ohe{D~OPo~N@~Od+~P!K~PrSC7~P)~N%]~M}~POaGD=IX~N;=Ih~N2t~NMG?!~N#fj/~OW8bN~N5~OS2~O^ZDPM~PLN?~P)~N*~N*b:~O^Y0U~P#~N-{Z&G~N@B?C~NLJ#AB/~Qc>~M\"S~N$8R/~N\'t|<7,J6y=~N8wLKG~MhZ:~Ph~N;EY-:v6~NL~NG~NM~P5~OmVP~Ohi~N4~N-\'~Q?{~N8U~OV~Oam~Oxz~NK~N<INmP3~OQ~P$L5>~MyFHd~NIQZc~MR`~NH?!5P~NJnWVH;h~P72~NM~NGe0M~O|~Og?RdIN~NJ94Q~OjQ~O_YW\"Lx~MJ~O]POv~N?+;i~OchSb6DpD=-~OYW*oS~Mp~N6~N8~Mc.;A^v~N4MoVZ;Z,<~MGOVY]~LKQ`O~O_y~NJQUpgv~O_d],Xo~Om~M5Q~OVas!~H;biGHtEJe~MnLIC~P2A^ov<A{7~O|~MxY\\I~P?35jVdj~Ol~OV!~P*w~Ox~N@~N9JJ<5l`;Mo{D,~MQ~M[G~O\\,Yw=e{^f2y]X#.Kr&Yef~OS~Ow~P[XwhtYZ~Q,~Og~O{nAp@~Oax~O]~OnJ~OY~Mw.~Oq&~OS~Ob~J^W&a~O],MKK~NH4~NO~N6M~Ob*@$~OuYKuu:X~P6~Ou7~MV5X~OP~OTdR>\\~N*~OQ0~PDP8~Mr~OoB:~Oi~OP5+6R;~N<]<Z~P7:~N,\\a;j+~MvTQ~OXn8+~OY~OP)5~N$@M!~N@_A\'sIFe~N/Ee~N8ca(W1~N6Mba~NL~N$4~N&Q~N=C~Oz?5~NG~Oia~MsY~OY#1Dg/Z(~OgXv#<Ol~N@^~OPr~OTU(~OnH5J7YJ-^FW~PO~O_V$&1H~N,O37p]9R9,S~N$<~O]~M`k1.~NJ~OPg~NCJZPf~OWa~O\\wM+tD~N-;vK)=w~N@~M}~P4kL~N?_)d&nexH<K(~NM~Mz~N?I~OVd~OW<~P2q~OR\\~Ok~ND~Mz!f?~OX^x\"u~NK~M|j4~PR+~NG~OS~NG~P+6s-VB*}W~O\\^/?q,~NJ<Q~Olp5J:Un+2n</Uw~N+-16>~Os1g?d~NNq67~Mh~N6;xB~OS5]r9V9.u2`iI6geCaR2b%j5\\80\'g*c|~OWA~N0!nwKK\\\\G9hTUnWGL~NO~Pf6~NG7irpT\\:][~OURS~NAz~O_:hd~NJZIh?m\"9SDAJCDfU_~OW>B@~N2SmdNV+dl/IJgPL~Os7avV\\~NC|FjF]g~OV6Gv~NB,M~Oi~N=I~NDIauEQ~NN\\D/C~NIgU:Um~O`wB~OjH]`e|g<o-VXeAnz*UL0~NN!2$=~P!S~M%w~NEjY*~NJbP~OqKf_KLE\"~O\\F&Q~NAB1~OS\"jE~OQmS@E~N\'8~P0=idsM&.]oh~O{PNo4AA1]/5h~OZT~Os~OW_mQ~N/AW19[q8?yqVJKT9X~PT~OoUQ~O_~N.]~N0~OV=T+Z\"~OlETdTMzQ2~Oamfw-<6U~P$c~NJ~NI~P/-U^2~OQ[=Mr:~OQ~N56Jb~N7~N?~OP~N3W~N=~NF:BOH~N\"i~NJ~N=~O\\`[!~N;D~O\\>~N,^[mJ?TV@~N2cS_~Oe|Y~OXS~O^i~OY}UD~OUDR8@~Q.mN0~NE*WL~O\\@jj`~Oo~P6ri~OnQ~Of~PFkQ*D~OuNf5\\pi~OWUW/N~OVVm*!_~M{~NK~ND~OPu\\FArl,ERd8~O]L~OoaY6H_5~N</QH1JzN)@19m~OQ~OqT~N#c3dpC~Ol~OT~O_~N9BT~P1qFJ#s_C^cQQK\\(FXA_~ND~N/qh(~N8#H~O_~NO~OTp:MQq~NIPth9oD>=tEKuVP~OPDP`9~P846WD?~O^~NNo~OXEg2V~MZ2^Ge,=6PFD`@]AP~O]~OT~O_L@cDj+7Kr1OR_KQJZY/_~N:\\-H=B<c~Oqo[P+bYFTVGH%i?XQGWwc3xJL`Z~OZ:GG.~N>vO\"3Df`KH=V.Lv1h[RR,~OSl+>b~O\\a@1|q66n2~OXGYRP^Z~NES?uX~OXCZJL~NFC)s~O`6c~OYO~O\\h1m~Oo%wCb5]29\"1.O^g-~Om~ND~NNZZ~O`\"kHBRZY*~OR9Y<$~NNjOkcT@c<q@~N=~OgRRM+6_~O^~OZ>~NE([K~Ox\"W~OzMxI~NBtE~N>~N@DRs~Oc*~OW~NGz~N8@ULi4BT~NA.Ib\\+?zm~N\"c(M~NO~NMA6~OV9)^;~PG=~N#$dBlQ./GQm~NJ~N<~N;~Orm~N;X~OoM~N6~N<k[>4ENNX>eVD^CB$~Oo~OTLroJF{~Od]E~MBa~Ok^\"~ND5P~N#G$xf3@~OgHC(bb)[~O|G%G/NyOkV:nL)~N22~NDP~Op;I~N?-=P4&PBblH75-~NEEI8~QS~Mx{W-N~N9PB~N@~OvM~NNX8\\n~NO~MY~Ow~P.XWC>~NF]EKJ7~Ok-M?D~M\\~Ob~OWK_b~NA!=_O<~OWv~N/BPW~P4;D~N&~NH~Oi~N91NEg:~N/~O{~ORJ|~MM~OR~NL2~N19d~NI~Pc:.[~NC\"VLm~Okvn~NG30~PI6~Q9Nhiq~Oj=~N;+d~NDE|=~O`s*OmN/(X<.~O[~NAIM>jH?g~O[8~N5Q3\\H9(~OkH4_B=I}zfA#~OtBJV/`7~NG~Otk~N9N~N+5Q%9EQMgZ~OY?TzJD+{3^l~O_~N0H9~PV~NOf?X?~MSX~P{-~NL~N&B_~N(XX~OkpAL~N._5~NF~Oy(QN;761-6a\\y~MqW~NA2z~N>>\"~Ot~P!P1WfS>O~OPQ~OQP-j8-D~N@~Oor&]:8~Oh_c,)V;h[@A~NKn`!g>k;~NAm~N4*P~Oako&!~M|wd]m~NF~Mz<G~NK<CiBY2~Of&EM~M}?U#!Y0~OSpD>XCU<=~MGDG~LzA?FPO\";T~O^G_35d~Ou~Og~OZ~NF9~N92~NJoI.~OR<1Um_~N8Sv]BV)WX~OYb~NA_N~N=KYBb[~NE@;7;b:~N8Q)n~O[~P(w?<HkcE~PsIU~P&9W4H-~Onc~N(0~On~O\\>QUJkY=~N#Dd~NIuUZb~N4~PF~OX~NI~Oj~NCA~N/<EV~N&d~N6\\~OP~N>p3PhU<~O`C<!3AE\\Sm1q~Pyrf%]rRF~OQZ4I~NJX;7{$@V&0gyBl4N.T`7<_[~Mv\'J`S-g~OR~O]9H;gD=~N,7~Qy~N39~PQF:Tct6)~M/N~P1;~ND%~Oo50~Mnb~L:~NKF~O}OPM=<$~M|A~Oz?^$?b]QEPi~Mn~OP~NDjH~MfRK86~P(\"Ag`6q@E]B~P%V@H<vL~NMAd=cr~NKx~N7l~NCi09o~Oc~OZ~PgNs~N\'1%x!regfJ3~OYm~N1.pW~P:~OZQ~NGt;,<Oe6~MH)X:~P8M73~N<$~NO37|G#*x~MgFX~NN~OU~NE~OS\'L~NL%GR~N5=~NL~P(~Ot~Oi~NJn~Osi~OvN~Oo~OT~N?~On~O`b]~N:keR~PuO~N3~NB~PT~OsL3~OT$Ss~OU~OW~Oc~NJ~Oa~P,u~NGO#zc`#~P!W~NL~NC{~NF-~N0tnWz~P*S~NB+~N@S~O\\~N:y4&uC-~N;!t~OP~Oe~Oe&q~OxN~OZ0ZZ~Oc~N=hx)*U?~OoU~N8~O|~N=~NL~O_%v=tm~NE~OT8W;0~OZS\'0~O^~PH~Oh$~OaoT~OZ~OQ~OcJ-~Oz~NM%~Omm~N>u>~ND~OeL1$~N=Fq~NJ~P^P~ND~O_~NKg+~NE~N8DJ~Odk%.~N;\\L~N4]?~NGE~N/}~N>q~NO8e3~P$q~NN~O_~N=;l}~Oa~On,~P))~N@D~P#R~P+~Oi~O[Ka~Om}~NG~NLt9~OU~Ou~Oz~Oo~ORzA**)6~OU:y~Oh^~M{~OTG}6<2~OeaG7PPDS]\\GL[,PDJ_An+_J=Ul76E~O[T?3lt[o;J]RVdlUKZV5~P*D7/~O[k\\OzVYLr^L<jlSD:>nYPE{GO@i=NAVFcP~OTA>>.Cn@h@4~OV[S2Ag^`xAs}NbGfb^<MfO.A@UK=mMNI\\eejg3mW>X*TiP1p~OXh\'{KMi__QBkaLl_2a\\LrbE3B[{<~OfK~NJa1jj8,GOw^HH:/L2TA6T7\\Dm1MJkldLd5.~ORaT_R]\'Kh~O`NzgGjP~O^MEHIN]ph~OYlkpK.JC[6jl~OU+act:V7nZ@wsTmV*0Icr8f~N4~P#1yA~OQ~MOb~O[fxf~NLM\"~N/~PWR4(E$ACJS+:~NGKZ*~N7`~PU9~ORZ\"D:~Odn~N%~NI~N9E.~NM~PsGx~N;~O}L~O]~N+~Oe~N9qL=SUje~OTz~N/%5#DP~PA~NG~O_]j5|Yh~NA;~Ok~NA~OV4P\"~NIx!Zh~NNE~OZA@V%.~O^$~OQ~P=~N8{.~NJ~OV~N#By7~NK~N*:2:Bio;S%k~Olc~N4x4~P+I^~N1~N>~Ou~Oo]%~NKZeKG~NE~P#~OS,>W~O][~N,~N8w~M0e~OVa]~Ot?c~O^/L~OeK~Of~P$l3~OWlM~Oa~P;~NC~Pa1~OX~OpmM~OT~OmbF?~NL|?Cf`\'}\"3sV~Mq~P!~M{V~NE0~NJ)=[pV~OjY$N+1~NOZlraRW\'33;~PK~OS|\"~OQ~NLM~OgA{~N+~N2);]?]9DX~MAI3~NL]C\'~M[i~MT#Gw~NKi~Oh~N2c:~Mhn8~N4b2~Om^~M{}^~O[f$~OTv%~Pe~Os~Ma-w~Ok~OTYORT~O\\`wa~N8~Oje*di(~P#[~NF(FG;(k%~N:.~Og~OZ~Onor>~NF2\"vs~N@~OUF~N6lS~M|~NK.P:l~P!*Bf3^~NG~OQGt~N?R~OeS~NMBJe~N.-~PIc=~OWdyNh+~N:32NY~N$\\fFg~O[_OF~O[R9o~Ot~OQ~N\"/a~N.->[~NBmf~NOuV=~N:~N?,)~ORc8~N!m@I9~N\"3T1~P0lJ:,C,QF~O`~N6^~N!DZ9~Oyn~OS*W`6dT;uWMn~NNu~Mx~N<_qy~P*~O`ed~Oo~OU*VHc0s~O`k#~Og]:P~N<~NBHW~NO~OUd>~M|eSRAk*l~P#hB~OPMOT@~NKu2~NJ[I<6[~NC:~On#d0dboIG9~NL~OV,P@hi~NK~NI\'jX/$)WAlo4GO_A>0,~O`naVG~O[~N3_ZHOG&G^YPM{feg~N4~ND~O[#~NI~OgK7~N/<~NB~NE~PlOQhA,HEP5=S<ea~Mq.w-]<~OXZlO9IV~N-pcT~OV?2~Mt?~OqTWW~N3W:w~NN~N8~NA~NNuD~OUg~N,X:M~N;~MzXHM~N)*=TL~P(F;{5~Oh]B~OS~OxXO;~OvGqN~Ov\'}~OV~NI{Y8PDfL:k_;C!\\~N2t;~O\\BMKO9\'~NO9rkEjGfp~M|b>o~OV\\hk~N4~M}~Oc<`!~O\\~NGf]~OaUt@~OPj^2.&~O`D~N4N5C~NI~OZSJZU`a~Mm~P9ns~N3>,Q~N;~O_~NOB2+~N\"A.4f7@~L}A~N7~NHMX~NEV~ND~OP~OedJp~OPulG~OgV#~OQa>M<MB*~PB~O`dF3~Oc~N@BhJ~O`Z~Od~Omu~N\"fui^~Mv`Wg~N5f~NHM4~NBDM&~OPk~Or~P%d~P5[`M~NC~OZ~Oe~LN~OUsL~Ocg.`<~NMa(~O]G~OS~OpfQ5Gp~OPDsV&9E1lC#~P0A~N7~Ox[Im~P$[7~OhLa~NLNL_lE~OP;}~N?~PZ~Orc\\~Og~OP~Mc~NDoK~NITg5~Ok~Oj]~Oc~P&E@YX~P!~NHy~OP_dVu!`,4;~OYACK~NGe64&G%x~N@x,~N?y9WP=d7T3Lg&:\\D`~N9_O7[~P`~O}yF~OYNt].SDd~P\'xE\\H`5~Mr+mV,_3+!gQ~OTCD/fXFLmsX<9*H|E~O\\:49[A*K;HaTnzI~NKaV>/D:4:Xch28Y]9;[M3;CQ@JP}eT4Q?F!QiPbo>~P((F~O^^~OSJ(D?I[~NMipShXmyD=Eh~OhKd5M@^!`Se-pEP#hhC@2Q!7G6k~OPV~OQ48AaM~N5]ES+XK>CDnnkTKHOVLMMiW<<?_~O[W3>O?EO-T[Y2HJRY+^#Yb)*7r~OR`k}LOaSn~ND?A,33B+S~OP&F7K?nTFPSRQZZJj<oW9^2~N4OwR.:XLWMdGTN~NF5a9dcL[fIP@?~O\\WXT78S~ORO~N*WA,UHC~Oq\'~NHg~ME,~Mw5p6~NLs~Oel5~OV~P#O~N0~N1~OT{~PUI36{~OR!~O[~O}dr~OW~P2~MVEV~Osj~NIsB<~Ok[~N,~NB\\o^~Os~NEEkQ#}~O_~NK~N.*a~N<n~OPr~Oj&O+kCB~Oe<W~O]~P)\"EC~N2!~NJ~LTX~MYHKLv~NLSM~OVx-,Z8~OZxC~MX^L~NNJh~N:`W~NFg[~N+~P62~P$~Oxx/E7Z*k)~N3~NFxTd,~N6peY,Q.~NAS`~N<N&~Oc~O[hvVHR~Ogdkn(,\\~P(y~N6[~OW~N?~O`G0~OXcH~N2#US~N>(Z~OVF~NO~NI~N\'X8[a7~MU\'Y^~M}pX@~OX(-Z_d~OU~NG~OlA^~O[i~N,~PEyMX~OluF~ND#`a~OP~P(t1Jy`~NO~OVI3~Os~NAei~Oj~OSL~OQ~OdZj~N#)No~NFC~Oim~N-5p~Ov~ND~NK~QKQ~NJ~O\\<~Ol\"~OZm]g~Ox~N&TUZ~P2~MI~OmQx~N,D~NLI!a~O\\G\"~M`w6~NKl~Oh~Ow~N<btY~Mh~Oh|[~OataDEe%~N<n-~O]rMMLMp~OSlx~P!~M9~P*gsY0HfX~OU*G~Oty*~Q10$a~Mpe~Or>D~Op~N:~P:~OaB~Ot~N9~Mq~NF2~MSm~NL~OV~N>#\'d3!(Y~P`~NEe~Oq~NI~PE<z~Od~N>Ac?7:P:~NKh~N8(~O`uA0~MnXM~ND~N4~Qk2\"~Oa~OZ~NI~N@FY~Oan~Md6Q?~OS~Owv`~P>~Mt~Om0\'~Or6_yN5~Ot~MlI~NNT3X~NB~N$G~N8I$O@~OrR:~N%~Oa[2J4Cc~OwP~NEk-\"]~Ow9~Q8]~NHU~M{~MN~NC~M^~Oh_a<ZOE(~Mw~MEc~OhR^Fb~Oys=~N/~OUB~NB~N6~NOe_WwI.?:wbr~O^~Op(=X~N&S.~NE~OYB~P5f~NKN~OT~NMW|8P~Od[`~OT2R9~NI\\jM|ze~Oa~M`9e~Obr~O]mCNeXt7[&RL~NMb{SB~Or?2b2zJ~Oeh~On~OPf5pvKk\'r~NBgWCg%KEN~NBU:6C~N9K,6~P$i~O|jT~OY~On/~OQ,~NDLYT~OTm0BQ~Mq~OW1*HC~ND!~PSOb~Mx@3M]~NIk~NNDY~N7~N9~NL?{8~ND+JK~OwHs>q~P1~NJf?Sck]ifI5R4,(T~NL0z*~N)6D@\\_~O{VU~N@Et>~MelE~NH~NH$`P~N>IB~PPY~MzbRkY-~OSVF~Og|~P4Vj\\#,RQO@{~ORV~N/~OY|O]{~OSzS?~N?uYODw>o~OXGX\'d~OZ-a>/[i`[RbU>OWaGjOmHDTPbR_MTYQW`APDbXRX=_M9bRD;^MVQ`~NOgzY4GaRCKLQALG~NESQJc\\mI0Ih?C^J[\\XFOEXX9hHRwZ]nNSh=N]AYPV]?REJ\\Ld]J;RPL[ZUKlOiGQ[RWCOUDZEcV.hUXbkG<:nGD^DCI`_>XO;BdW:SARe`XY5i:PlFh1~N>jtbWEO[CO5XjSRMp;O_d^TA=?LaD]]<LMENAeTQ>NDG/$3WZ_^UV?BB:WFDUeYbFAFNV];FO[\\WHGm~OeeemW8%i~Ol4~P8smN@\\Ho>el2VY~Ob~O]~N;~N9~Mx~Od~NN&T~N4~OU)~N0a@~ND~NG~OS^=/~O^Nh_y~N5*~OT_~N#~NO62NI0~OW:@\"~Opr~OY=z:D~NB*vVV]Z~OW47y9:y~OSk~P&\\U~P-3~O^p~NIBR~OVk8?~NA=j~NE5Q+tT~NB-aD~N?vvpjPo~Oj$u24a:u//p~NG~Oq]m~O^@`s}#~Mp7~Oy^I~OV~NM.eIp~NE~N/[~N$2~Ok0Ji<Ao{l~NL9~O`~Mbj~On~NCw1[z~O{~O\\mLHz~Outn\\~Of<~Od~OS~OtiGcjow._$1i~N9~OTw@/F~NO~OW~NFya\'~MCe~NI~NH~N&P~NB~NG/{~O]B[6@~NI>S~NH&zst~OP3_BM~NO~OW~NI`\'rD~OW~N&;~OPnd~P8A8L|j~OX~N5QP~OjFG~Oi~NIdu]d=vN~NKv~N0>z+~P1\'~PJDu<=Q&~Ji~OX~OZ@E0JZj~NI`~O^D(~P)~Oz~OQ|~Pj1~NLsM~NN~O{3R_o*kK{#>x\'~Oj~PeAf$4W%D>2je~N0dt3E<~N/~Q*>~NGM~MY~NL~MN~Mxb2V~N9Bcc`eX<~K.DU`C/`kLD~O{lZ3~MFFSa<_~OV~Mg|L~NIT~Oh++~M{~OWV~OQsI@Y@N~NJ0~PNdI.7~MiSI~Mq~O]?Qv~NHhW~PEN~Ozn~NK~L)JU\"~N<O#W;~N5FO~Oe5:Sd~OeXFm~O^~L`dJ~Oj~NB~MhkHc~R<H\\\'?Oi:D~M}f=b3~M9aekPk:~O];R*Y~OWW~QUy~O[b*8e~NA~NF~O_u<~MiSW~P&gW#~M{e3=ArxS~N>2aB#oYMQL-??W~NO~NJ8~OU`_]<~N):R~N?]L9:2\\!.}PrGM~O`~N;0JAlK$~OSbo~Pi~OQ.<~M{V[Pnm~N?7VrPw\\4c~Oh~OT_WePK43Rj%U~OS;Ve.%~NOB~NO[d~OTL~P*x1MN~N0c#]g`A~OQ~PruS,~N?1~M}Qw55#9`9Td8A0~N0~O]8~OUx-/;~OP~Ofo~ND~M`:~PHg81J~OP~NK5E~OT~NK><L~NL`CA6+kYQM~Lg`6\'`21jR9\\cQ.CZ~NI~PRi4~Oc\'J;]~Mn*QP\\m[Z@E~Oj96~Oi~PUI?B=~PNgPKn:~N;oB8~NIQ;~O}CL+\\N&s~N0S>i?`<~P6~N?<D?9CXFdsnfmyS/fweP~P#~ObCl7~OVF]~NA`VEJH9b~N@IFnMHl@b?Op~NN~NA~N9W[~NJD`zTS~OX~N:3M~OS^~N@G~OXKJ4~P\"H~N@Jo\'sC~Om>i~NFBpN@`eCY7~Ob~N155-QY~OV~NI~Oa3~Oix.&T>nQ]MKwV2j~NF~OhI~NHMr?~M[nDzbV~O\\~OR~N@~OZ1Cp2~NGNCa7_bs<#/|x6~N2<<rC~O]&$rDG\'8P~NE,bQ~OS~O]~NB;utm5mEhH~P.Jo~P%]rylz~Oh]i~OU[Qn_V~OV~O]T~OjD~O[R~Of]l~P!+<>\"~OUp~NChI6~P)~NOnoO~Mcj2Jm,~N1~MKU|ARu`~NCOi[/p6hb~OZr^W/P_?X[&.bDKEP=~NE~OxgN$DRN[~Oon+KG~MTw~My[K{~N,U~N\'~NE~Mq~N=]kp~NK+F\\%~OTW~NF(~NK~N9s47~NK~P\"~NLofp~N<JA}~OR&aQR~N1pfjv_7K5gT~Pj9\\T@w[gy3~L8~OPI~OhN~N@R~O_~Mp~PMgHlKH<O~O^m~OnBC~Q*|<m}7NT+~NEz2>~OZ~Pbz~OZ~NFr,&R~OmEN%0~Ol%d<=L/tDC(~NG$>q~M9~OzqlF~N?@~Q$nLpQ~OV~NI*~LN{Z#)fut~P8U)J~M7F%hm8\",~OVQi~MN<;}~F#o~OR>~NA~P-~Oj/~NLsPOcIt~OqXIDM~OjE{#~NEV~Mt~OUh8~NEM~LoQ-~NC~N1<<e[\'N[Au~NBR~M[grfuO~NI19Wg~Ou~N-~NCOy~Mw9~PJT)~MvF^Sm=~NMj;@yMZf;F|>&R\\Ei*P~P/K~NF~NJh0JN2~ND6(oGMM~OYO7~OoM._ew~M}m8J6FgQZ22~N5_P0C~P\'~OS~O]~N.T.)2fnwMSkW9l-(Y~OjE~OU^~OS]&#}~OS`Zof~NL=~P74X~NJo%om5U,~NOS~OY~PYFH>Xv=~Ol_|t~NIdWV.~NFRH3uCp~Og|d1\'7~NC)[~OfXM~P#bFNC;~NK4r>q~N@0m~NL0~OP~NK-~O^~MlqTC}~NG\'ShsCW5~ORpHa*)C[~O_~Ox:A?PBHL~OeEQLmTRf!`w&?!9/~P/~O`U*mPH[BVFB7s~Mc~NGfC.e|MD~OvHDL9V~O`q2y~Oh3~OZU(~NKMQMc~N>-~O^FM~OrFk\"D@Py~N;?JE~OS~P.8C&t_~Oc8eA~OS~OW;~O]f<6~M|a$G]|K\"wQe~N\']~N!~N@$yq~OTw-V+s~OrZ)~NF:u~N4UVC{W\\*~Od0,NCd+~OSF,k5?~Oe~NG<3W~NO0LlLInO*0~OW~OP,LT~Oag(B30~NA]K~NIln!~NIOcnd8c;EI~OR]~NJ~N<kd~NC_@QnF}^_wV3~OR[9~OV5~Oz~NEW~N<~NOjNe4nGNoYM}oepXo:K~Ou~N7[~N4~OgN?~OY;Es~O^3\\cApN~NMZIn~N\"~NNDZZN~OT~N0u~ORx.h\\Vk~OVb+~OPW~Oc~OSV}T_\\T~Op|[~N\"(~NNV,~OdSTiH^wjc.:U6v~N=~OWj.~Ox~O^\'W~OX[#~OSjLO~O^Ak$RT~P0Hrrj~NCljD~Mz~M8YgR\'udg/J[~N<?~NH<~O\\)/~PB~Pi~McSrq~MS~N4_~N7x~NNm~N+~N8~N>~Q!~NO9~NEY\'n\'~OU~NCe0~OR]/)Z~OQ~OQ~N?~NM~NC+nw~OX~OP~On`~Mi~Oh%O~Oj#\'~PJd}(6~M>M{~Q$?34~Oyi\'~NGt~N/6.~OW_2Ap~Om~M-=|~Mxd&u7.~Q!wJpRQ2528$y&`i~OgH8~Md~MlnPQyn%~Mss\'=~OS~M`~N+I}tD2@~MYNda\\|[TH~N6~On~Ok;~OT;[@~P.fB,~Or~NO~QV@R~OhO~Oa~Mv`~N%~N8Q1\\rR)8~NK~NLY~O`W<~M1~OP~MrU~L8_(.0C~OS~NM]=w+*+8P6~PV\\4IW8Q~Ozw~NIL~N#TZMG~NC~N8[g~Oas6OsDt6~P==~NHWB6IYRXWVgY~NH$ka,JOt.~Ok=V~O`AxX~O|$Sn~OW<~NE5~OVsTA5~NH~O_xns~P*~O]w~OQ)bh~NH~N+@K~N>h~OP~Oi9xGj^X~N7~Oa<~N4~OW~OU>X~OoO5x~QD+k&OYd{`l~O^x\"UA~OsUwunv[ixJ8g]~NL_g~N4VH~Od~P%TX*V_W6[8_gUh0~NJH~OX;6~Og:dL~N,~O\\~P1&k~P?wtV1Ek~MnmIBK7]q:~Pp~P,~NFmni\\f8Cbh8FQ?cDCz~NJmK~OSUn.~NN~OYNV~OQF+n5RcZVjQyRSA~O\\-Z?@~OQ~OY~OT~Oc:xkBB~OR9=DQL~N=^L~N>l\\L{~ObzdmVPXM]uQ~OT~OU~N<]d@v:<$~OQ7H~OR<\"7lC~OX2kh{Df~OX~OQKr#~Pm~N2~MP~R.~M2~M>~PC~MM~P;~Q\"~Pv~Lm~Pg~P-~Lk~Q4~Pf~Q#~MR~Pz~L7~Q\"2~M}~L3~QE~N-~Kn~MY~P[~Lf~P{~PK~N3~N\"~Q^~M\\~M|~P:~M}~P*~PY~PK~L}~P[~Oh~M:~Q!~P9~Pf~N,~P?~L^~P@C~N5~Lo~Ph~N/~LU~MP~O|~M$~PFDFdB:aDPN9ZTI<acVGdc@fVA>faePDDQ=RaGHR8LUA]a^PMc;XW@X<?RfDD]c@EH~P\'~N?~N.~PP~N1~N@~Oq~N.~O|~P$~O|~M\\~P!~OQ~Mc~PE~Oi~P+~N0~On~MG~P5C6~Mi~P4~NJ~M<~M}~P%~MM~P(r@~NO~P=~MC~N?~P8Z~OQ~Oj~O\\~N0~Oh~Oh,~P-w~O}.L~Mk~Oy2#~N%~OqY~Mr\"t~N/~O}~PB?9~P9~M;~N%~Pz~N7~On~PN~OP~N-~Pi~P\'4~Om~OQ~O}.l~Md~Ov~NE~NB~MW~On.~MH)t~N-~OzdQSPhKJ<Y6?[Z`r6:KUQa^B^p4GqONGRuP*fGAN>FpYCtnPiXP/d>h_:$SW(PpHhNg~O_~N2~OQsD~OYH1.~Oi/[|\'=~NJb~NH~On~NLbm~On~NF_~OspB~OS\";ISZeA:Q[cHMFb\\LbX;8Ub;G@LD`WdNL~NL~P\"~P6~MA~PA~P05~Oj~N%~Mj~Mk~PN~N7T~PJ~Ms~N+~N3~P*~N(~P}~Mqi~P2~Pu~Mn~P*~Q!~Ot~N>~Pq~N%*~Om~Og~Mm~Oq~OmMu\"~N;&~P61B~P.~N+%~NOu~NA~PL~NHwz~P0~NI~O^~P>~Of/~P>~N=~NH~P$~P\'~MS~P2~Og7~Od~N4~N7~Mt~Pd~NBY~P4~Mo~N0~Mv~P!~N7~PK~N#^~P\"~PY~N#~Ol~Q*~Oc~N%~PW~N14Z6?|S2;Y=jBHD;MJFFn@RMk_HgQHFOyDD]9VW?^:<F;L[YGX>9K8hGi9\\T[HX<D*x~Ob~M}~Ok~OkK~Oc\"~N:~NC~Oq([~Ow~N?!5j~NA~Ox~NGR~OX~Ov~N:~O\\~Ov~OS~N@~Oh~N?~PG~N3~Mi~QA~Mc~Mc~P.~Mc~P#~PJ~PL~M$~PI~Ov~MP~Py~P<~PQ~Mt~PM~La~PTI~N0~Ls~Pn~NH~LQ~Mr~P@~Ls~PLo#/~P$~NM\'|~NOu~Of~Og~N4~Of~OS~N;~Ooz~Os~N<~Ob~Mt~OXN(~N,~OoG~Mu~NEn~N+~Ot~NF~PO~Pm~Lz~P_~PS!~P1~N5~MI~MP~Q0~N)A~P|~MD~Mb~Me~P&~MV~QK~Mxx~PU~QV~M=~PK~Q[~P8~Mo~Q#~MU\'~PG~Pm~LV~P_~PI2~PD~Mi~M@~M_~Q3~N4O~Pe~ME~MX~M_~O}~MT~QW~MW~OW~PZ~QR~MP~P<~R\"~PH~Mi~QA~MD~N(~P{~Q(~M/~QQ~Pm~Pz~P*~N/~M[~N;~Pn~NKf~Q,~NG~N(~M\\l~M0~Q]~N1$g~Qn~N4~Q<~R;~OS~N,~Q6~N/~OiVY~PG~M(~N1~PK~NA~Ob~Ohc5~PT~P4=~OeS~P(DB~Ml~OV~NB~N@~Mf~Oh5~My=Z,H~Ok#~N5~Po~N%~N4~OX~N$~O]~P4~P#~Mc~Oxy~N%~P?~O\\~P)~N$~P$~M^~P*R3~Mc~Oz~N;~M<~M}~Oq~M^~Oqu02W=>Q@fvn=Mr:u~OQc9j(W7I~NLW]~NODo%{PGao:6^;Hnf<vrKJVl<Z7e44Lk@6IH8h~N>~PE~Pb~LP~Pa~PF5~P5~M`~M7~MD~Q4~N)S~P|~M&~MU~Mj~P<~Mb~Qg~M^~O^~Pe~Qj~M/~PK~Qk~P[~Mk~Q?~MTMHZPZeBJXM_:\\]@QIbNRD_GH`ZdCXb@NU~Oz~P\"~MY~P|~PJ~Ou~QH~N<~Mt~N?~P<(m~Pc#~N?~N.~OT~MK~Pb~NHFh~Pf~N2~P_~Q$~Oo~NO~Pf~N1~P[>v~Qv~KY~ME~RJ~Me~P\'~QFz~Mx~QT~Q\"~N9~PY>~QJ~N@D~Lt~P(~M;~Ms~K}~PA~N,~L?~Mvz~MdtEl~OZ~NA~OSe+m%~N8*~OZ:H~Op\"~NA.~O]~NI~P#-V~Og~Oj~N>~OU~Oi~O[%~P#~NO~Ps~Da~FlHU~IB~Jb~IP~OP~N!~QPI~U?~S*~T_~N9A~RJ~Fq~KF~W?~W[~Ww~JJ~IW~MR~N!~X!f~UU~EC~R!";
std::string biases_string = "zBQ]a5~N2_TAW~Le~MxI~OP~Mp>9HNE`Hh6e2~L_~P\\~OxV~N@A5~MMR:ZP9X~PO@,TNL~LYl\\,t+i~N\"N~O[s~NCAP<~N7E~OPK~LVuoEdE;~N&_U[!~Mr#2d]J~P\"Mdqc~PGxb~MWIX~Oqo~NO?Y(C~O_~MP80~NE~N(IuK,~N?93xaG\\1~Ok|R~Mb~N,~P9@~Og&/L/a;~MaR~OP~Ml[TM~OsG?~MU~N;N~MW>C$4~N<~MV9~M6\'7~Mbd1~NMK=~PPqg67CXX~L]c~Pbn~Oc4=h~OX~ND.~MwO~M^]#;~N\'?(~P+;<~N\"XKM~PX5@~L_~MWBQ~P4~P-c~Ma2X~NO~NO}O=9~M&E~MOlgx~Pa~P1~OW~O|1]1[+:!+C~N2{p~NAbY~N+CQWF~NL[0PPc+6~O}CN~LH~PF;k?~NB>~PE,3z~N*#P~M[,~PJ~Lr(~OT~P+*~NA#~Ok~N;q2~MTg~P}~PWA9~P$wSK`iR~Ou}~OmIK~OU~PZ~O^~P8~O{~M}A~P+oS~P;4~NJOzX";
int main()
{
    int idx = 0;
    for (int i = 0; i < WEIGHTS_SZ; i++) {
        char c = weights_string[idx++];
        int x = c - '!' - 46;
        if (c == '~') {
            c = weights_string[idx++];
            x = (c - '!' - 46) * 93;
            c = weights_string[idx++];
            x += (c - '!');
        }
    
        QUANT_WEIGHTS[i] = x;
    }
    
    idx = 0;
    for (int i = 0; i < BIASES_SZ; i++) {
        char c = biases_string[idx++];
        int x = c - '!' - 46;
        if (c == '~') {
            c = biases_string[idx++];
            x = (c - '!' - 46) * 93;
            c = biases_string[idx++];
            x += (c - '!');
        }
    
        QUANT_BIASES[i] = x;
    }

    int id; // id of your player.
    std::cin >> id; std::cin.ignore();
    int board_size;
    std::cin >> board_size; std::cin.ignore();

    Experimental_Board* board = nullptr;
    // game loop
    while (1) {
        char grid[8][8];
        for (int i = 0; i < board_size; i++) {
            std::string line; // rows from top to bottom (viewer perspective).
            std::cin >> line; std::cin.ignore();
            for (int j = 0; j < 8; j++) {
                grid[i][j] = line[j];
            }
        }

        if (board == nullptr) {
            board = new Experimental_Board(grid, id);
        }

        else {
            std::string opponent_moves;
            std::cin >> opponent_moves;
            // std::cerr << opponent_moves;
            int tmp = opponent_moves[0] - 'a';
            if (tmp < 0 || tmp >= 8) {
                board = board->advance_move(-1, -1);
            }
            
            else {
                for (int i = 0; i < opponent_moves.size(); i += 3) {
                    if (i > 0) {
                        board->advance_move(-1, -1);
                    }

                    int x = opponent_moves[i + 1] - '0' - 1;
                    int y = opponent_moves[i] - 'a';
                    board = board->advance_move(x, y);
                }
            }
        }

        int action_count; // number of legal actions for this turn.
        std::cin >> action_count; std::cin.ignore();
        for (int i = 0; i < action_count; i++) {
            std::string action; // the action
            std::cin >> action; std::cin.ignore();
        }

        auto start = std::chrono::steady_clock::now();
        auto [x, y] = get_best_move<Experimental_Board*, float>(board, 0.147, 1, 1);
        board = board->advance_move(x, y);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        // std::cerr << "True Elapsed Time: " << elapsed.count() << std::endl;
        std::cout << "EXPERT " << (char)(y + 'a') << x + 1 << std::endl; // a-h1-8
        // std::cout << (char)(y + 'a') << x + 1 << std::endl; // a-h1-8

    }
}