import React, { useState } from "react";

const Cal = ({ show }) => {
    // 상태 변수 정의
    const [hourlyWage, setHourlyWage] = useState(0); // 시급
    const [hoursWorked, setHoursWorked] = useState(0); // 일일 근무 시간
    const [weekDays, setWeekDays] = useState(0); // 일주일 근무일수
    const [extraHours, setExtraHours] = useState(0); // 월 연장 근무시간
    const [totalPay, setTotalPay] = useState(0); // 총 급여
    const [calculationDetail, setCalculationDetail] = useState(""); // 계산식 설명

    // 총 급여 계산 함수
    const calculatePay = () => {
        // 주간 총 근무 시간 계산
        const weeklyHours = hoursWorked * weekDays; 
        // 주급 계산 (시급 * 주간 총 근무 시간)
        const weeklyPay = hourlyWage * weeklyHours; 
        // 연장 근무 수당 계산 (시급 * 월 연장 근무 시간)
        const extraPay = hourlyWage * extraHours; 

        // 주휴수당 계산 (주간 근무 시간이 15시간 초과 시)
        let vacationPay = 0;
        if (weeklyHours >= 15) {
            vacationPay = 8 * hourlyWage; // 주휴수당은 1일 소정 근로시간 8시간
        }

        // 총 급여 계산
        const total = weeklyPay + extraPay + vacationPay; 
        setTotalPay(total); // 총 급여 상태 업데이트

        // 계산식 설명 문자열 생성
        setCalculationDetail(`총 급여 = (주급: ${weeklyPay} 원) + (연장 수당: ${extraPay} 원) + (주휴수당: ${vacationPay} 원) = ${total} 원`);
    };

    return (
        <section id="Cal" className={show ? "show" : ""}>
            <div>
                <label>
                    시급:
                    <input
                        type="number"
                        value={hourlyWage}
                        onChange={(e) => setHourlyWage(Number(e.target.value))} // 시급 상태 업데이트
                    />
                </label>
            </div>
            <div>
                <label>
                    일일 근무 시간:
                    <input
                        type="number"
                        value={hoursWorked}
                        onChange={(e) => setHoursWorked(Number(e.target.value))} // 일일 근무 시간 상태 업데이트
                    />
                </label>
            </div>
            <div>
                <label>
                    일주일 근무일수:
                    <input
                        type="number"
                        value={weekDays}
                        onChange={(e) => setWeekDays(Number(e.target.value))} // 일주일 근무일수 상태 업데이트
                    />
                </label>
            </div>
            <div>
                <label>
                    월 연장 근무시간:
                    <input
                        type="number"
                        value={extraHours}
                        onChange={(e) => setExtraHours(Number(e.target.value))} // 월 연장 근무시간 상태 업데이트
                    />
                </label>
            </div>
            <button onClick={calculatePay}>계산하기</button> {/* 총 급여 계산 버튼 */}
            <h2>총 급여: {totalPay} 원</h2> {/* 총 급여 출력 */}
            <p>{calculationDetail}</p> {/* 계산식 설명 출력 */}
        </section>
    );
};

export default Cal;
